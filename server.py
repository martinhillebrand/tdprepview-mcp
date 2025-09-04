import os
from typing import Any, Optional, Literal, Dict, List

import httpx
from mcp.server.fastmcp import FastMCP
from utils import suppress_io_and_warnings, create_db_context, close_db_context
from helpers.sklearn_data import load_sklearn_df, upload_df_to_teradata
from helpers.autoprep import create_and_fit_pipeline, generate_and_save_plotly_figure, deploy_pipeline_as_view
from helpers.modeltrain import train_random_forest_model_payload
from helpers.modeldeployment import deploy_onnx_model_to_teradata

with suppress_io_and_warnings():
    import tdprepview
    import teradataml as tdml

import time
import logging

# Initialize FastMCP server
mcp = FastMCP("tdprepview")

HOST = os.environ["DB_HOST"]
USER = os.environ["DB_USER"]
PASSWORD = os.environ["DB_PASSWORD"]

# MCP Tools

@mcp.tool("get_dummy_data_upload")
def get_dummy_data_upload(
    dataset: Literal["iris", "diabetes", "linnerud", "wine", "breast_cancer", "california_housing", "boston_housing", "titanic", "adult_census"] = "iris",
    schema: Optional[str] = None,
    table: Optional[str] = None,
    if_exists: Literal["fail", "replace", "append"] = "replace",
) -> Dict[str, Any]:
    """
    Upload a scikit-learn or OpenML dataset to Teradata using teradataml.copy_to_sql.

    Available Datasets:
    ==================
    | Dataset          | Type           | Size          | Data Types            | Description & Characteristics                           |
    |------------------|----------------|---------------|----------------------|---------------------------------------------------------|
    | iris             | classification | 150×4         | all numeric          | Classic flower species, clean, no missing values       |
    | diabetes         | regression     | 442×10        | all numeric          | Diabetes progression, continuous target                 |  
    | wine             | classification | 178×13        | all numeric          | Wine quality, chemical analysis, 3 classes             |
    | breast_cancer    | classification | 569×30        | all numeric          | Medical diagnostic features, binary classification      |
    | california_housing| regression     | 20,640×8      | all numeric          | House prices, larger dataset, geographic features      |
    | boston_housing   | regression     | 506×13        | mixed numeric        | Classic regression, various feature types               |
    | titanic          | classification | ~1,300×12+    | mixed cat/numeric    | Passenger survival, missing values, names/classes      |
    | adult_census     | classification | ~48,000×14    | heavily categorical  | Income prediction, education/occupation/marital status  |

    Parameters
    ----------
    dataset : {"iris","diabetes","linnerud","wine","breast_cancer","california_housing","boston_housing","titanic","adult_census"}, default "iris"
        Which dataset to load. See table above for detailed characteristics.
    schema : str, optional
        Target Teradata schema/database. If not provided, defaults to the current DB user.
    table : str, optional
        Target table name. If not provided, defaults to the dataset name.
    if_exists : {"fail","replace","append"}, default "replace"
        Upload mode:
          - "replace": drop/create table and insert all rows (overwrite).
          - "append": append rows to an existing compatible table.
          - "fail": error if the table already exists.

    Behavior
    --------
    - Loads the selected dataset as a pandas DataFrame (with target where available).
    - Adds a sequential integer primary key column named `row_id` as the first column.
    - Uploads to Teradata with `primary_index="row_id"` via `teradataml.copy_to_sql`.

    Returns
    -------
    dict with:
      artifacts:
        - input_table: fully-qualified table name (schema.table)
        - dataset: dataset identifier
        - rows: number of uploaded rows
        - columns: list of column names (including 'row_id')
        - schema: resolved schema
        - table: resolved table
      log: short status message
    """
    resolved_schema = (schema or USER).strip()
    resolved_table = (table or dataset).strip()
    if not resolved_schema:
        raise ValueError("Resolved schema is empty. Provide 'schema' or ensure DB_USER is set.")
    if not resolved_table:
        raise ValueError("Resolved table is empty. Provide 'table' or 'dataset'.")

    df = load_sklearn_df(dataset)

    create_db_context(HOST, USER, PASSWORD)
    try:
        upload_df_to_teradata(df, resolved_schema, resolved_table, if_exists)
    finally:
        close_db_context()

    return {
        "artifacts": {
            "input_table": f"{resolved_schema}.{resolved_table}",
            "dataset": dataset,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "schema": resolved_schema,
            "table": resolved_table,
        },
        "log": f"Uploaded {len(df)} rows ({len(df.columns)} cols) to {resolved_schema}.{resolved_table} with primary_index=row_id (if_exists={if_exists})."
    }


@mcp.tool("create_ml_autoprep_pipeline")
def create_ml_autoprep_pipeline(
    schema: str,
    table: str,
    non_feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create and fit a machine learning preprocessing pipeline using tdprepview for a Teradata table.
    Do not use this twice, i.e. do not apply this on a view that already is the output of a preprocessing pipeline.

    Parameters
    ----------
    schema : str
        Database schema name containing the table.
    table : str
        Table name to create the preprocessing pipeline for.
    non_feature_cols : List[str], optional
        Columns to exclude from feature processing. These are typically:
        - Primary keys (e.g., 'id', 'row_id') 
        - Target variables (e.g., 'target', 'label', 'outcome')
        - Metadata columns (e.g., 'created_date', 'updated_by')
        Important: The LLM should identify which columns are NOT features to be processed.
        
    Behavior
    --------
    - Creates a tdprepview Pipeline using auto_code heuristics based on table schema
    - Automatically determines appropriate preprocessing steps for each column
    - Fits the pipeline on the specified table data
    - Caches the fitted pipeline object with a unique ID for later use
    - Returns pipeline ID and the preprocessing plan (pipeline steps)
    
    Returns
    -------
    dict with:
      artifacts:
        - pipeline_id: unique identifier for the cached fitted pipeline
        - input_table: fully-qualified table name (schema.table)  
        - pipeline_steps: list of preprocessing steps planned/executed as str
        - non_feature_cols: columns excluded from processing
      log: short status message
    """
    if not schema or not table:
        raise ValueError("Both schema and table must be provided")
        
    resolved_schema = schema.strip()
    resolved_table = table.strip()
    non_feature_cols = non_feature_cols or []
    
    create_db_context(HOST, USER, PASSWORD)
    try:
        pipeline_id, pipeline_steps = create_and_fit_pipeline(
            schema=resolved_schema,
            table=resolved_table, 
            non_feature_cols=non_feature_cols
        )
    finally:
        close_db_context()
    
    return {
        "artifacts": {
            "pipeline_id": pipeline_id,
            "input_table": f"{resolved_schema}.{resolved_table}",
            "pipeline_steps": pipeline_steps,
            "non_feature_cols": non_feature_cols,
        },
        "log": f"Created and fitted ML preprocessing pipeline for {resolved_schema}.{resolved_table} (excluded {len(non_feature_cols)} non-feature columns). Pipeline cached with ID: {pipeline_id}"
    }


@mcp.tool("save_pipeline_sankey_file")
def save_pipeline_sankey_file(
    pipeline_id: str,
    filename: Optional[str] = None,
    height: int = 800,
    width: int = 1200
) -> Dict[str, Any]:
    """
    Save a fitted ML preprocessing pipeline Sankey diagram as an HTML file.
    Always print out the file path to the user as hyperlink.
    
    Parameters
    ----------
    pipeline_id : str
        Unique identifier of the cached fitted pipeline (from create_ml_autoprep_pipeline).
    filename : str, optional
        Custom filename for the saved file (without extension). If not provided, 
        uses pipeline_id as the filename.
    height : int, default 800
        Height of the sankey diagram in pixels.
    width : int, default 1200
        Width of the sankey diagram in pixels.
        
    Behavior
    --------
    - Retrieves a fitted pipeline from the server cache using pipeline_id
    - Generates a Sankey diagram showing data transformation flow
    - Saves the interactive HTML visualization to the exchange directory
    - Returns the absolute file path for Claude to expose to the user
    
    Returns
    -------
    dict with:
      artifacts:
        - pipeline_id: the pipeline ID used
        - file_path: absolute path to the saved HTML file
        - filename: name of the saved file
      log: short status message with file location
    """
    if not pipeline_id:
        raise ValueError("pipeline_id must be provided")

    file_path = generate_and_save_plotly_figure(pipeline_id, filename, height,width)
        

    
    return {
        "artifacts": {
            "pipeline_id": pipeline_id,
            "file_path": "file://"+str(file_path),
            "filename": file_path.name,
        },
        "log": f"Saved Sankey diagram for pipeline {pipeline_id} to: {file_path}"
    }


@mcp.tool("deploy_pipeline_to_database")
def deploy_pipeline_to_database(
    pipeline_id: str,
    input_schema: str,
    input_table: str,
    output_schema: str,
    output_view_name: str
) -> Dict[str, Any]:
    """
    Deploy a fitted ML preprocessing pipeline as a database view for production use.
    
    Parameters
    ----------
    pipeline_id : str
        Unique identifier of the cached fitted pipeline (from create_ml_autoprep_pipeline).
    input_schema : str
        Schema name of the source table to be transformed.
    input_table : str
        Table name to be transformed by the pipeline.
    output_schema : str
        Schema where the view will be created (must have appropriate permissions).
    output_view_name : str
        Custom name for the output view. Needs to be set by LLM, something like {input_table}_prepared
        
    Behavior
    --------
    - Retrieves a fitted pipeline from the server cache using pipeline_id
    - Uses the pipeline's transform method with create_replace_view=True
    - Creates/replaces a database view that applies all preprocessing transformations
    - Caches view information in MCP cache
    - Acts as a "deploy button" for production deployment of ML preprocessing
    
    Important Notes
    ---------------
    - The view will be created/replaced in the database - ensure proper permissions
    - The created view can be queried like any table: SELECT * FROM output_schema.view_name
    - All preprocessing logic (scaling, encoding, etc.) is embedded in the view SQL
    
    Returns
    -------
    dict with:
      artifacts:
        - pipeline_id: the pipeline ID used for deployment
        - input_table: fully-qualified source table name (schema.table)
        - output_view: fully-qualified deployed view name (schema.view)
        - output_schema: schema where view was created
        - output_view_name: name of the created view
        - column_count: total number of processed columns
      log: deployment status message with view location
    """
    if not pipeline_id or not input_schema or not input_table or not output_schema or not output_view_name:
        raise ValueError("pipeline_id, input_schema, input_table, and output_schema are all required")
    
    create_db_context(HOST, USER, PASSWORD)
    try:
        full_view_name, output_columns_names = deploy_pipeline_as_view(
            pipeline_id=pipeline_id,
            input_schema=input_schema,
            input_table=input_table,
            output_schema=output_schema,
            output_view_name=output_view_name
        )
        
        # Extract view name from full qualified name
        view_parts = full_view_name.split('.')
        actual_view_name = view_parts[-1] if len(view_parts) > 1 else full_view_name
        
    finally:
        close_db_context()
    
    return {
        "artifacts": {
            "pipeline_id": pipeline_id,
            "input_table": f"{input_schema}.{input_table}",
            "output_view": full_view_name,
            "output_schema": output_schema,
            "output_view_name": actual_view_name,
            "column_count": len(output_columns_names)
        },
        "log": f"Successfully deployed pipeline {pipeline_id} for input table {input_schema}.{input_table} as database view: {full_view_name}. Processed {len(output_columns_names)} columns."
    }


@mcp.tool("train_random_forest_model")
def train_random_forest_model_tool(
    output_view: str,
    target_column: str,
    model_type: Literal["classification", "regression"],
) -> Dict[str, Any]:
    """
    Train a Random Forest model on a preprocessed database view.
    
    Parameters
    ----------
    output_view : str
        fully-qualified deployed view name (schema.view). result from deploy_pipeline_as_view()
    target_column : str
        Name of the target variable column in the view.
    model_type : {"classification", "regression"}
        Type of machine learning model to train. You decide based on target var type
        
    Behavior
    --------
    - Uses cached  information from the pipeline and view creation to identify feature columns
    - Loads data from the preprocessed database view
    - Splits data into train/test sets
    - Trains a Random Forest model (no hyperparameter tuning)
    - Converts trained model to ONNX format for deployment
    - Caches the ONNX model with metadata for later use
    - upload the model into a teradata Vantage Table "model_table"
    - Returns comprehensive performance metrics
    
    Important Notes
    ---------------
    - The view must contain the target column specified
    - Feature columns are automatically identified from pipeline cache
    - Target column is excluded from features automatically
    - Model is cached as ONNX format with single input tensor "features"
    - returns evaluation report depending on the type of model
    
    Returns
    -------
    dict with:
      artifacts:
        - model_id: unique identifier for the cached trained model
        - model_type: same as input
        - target_column: same as input
        - metrics: comprehensive performance metrics dict
      log: training status
    """
    if not output_view or not target_column:
        raise ValueError("output_view and target_column are all required")
    
    if model_type not in ["classification", "regression"]:
        raise ValueError("model_type must be 'classification' or 'regression'")

    
    create_db_context(HOST, USER, PASSWORD)
    try:
        model_id, metrics = train_random_forest_model_payload(
            full_view_name = output_view,
            target_column=target_column,
            model_type=model_type
        )
    finally:
        close_db_context()
    
    # Extract key performance metric for log message

    
    return {
        "artifacts": {
            "model_id": model_id,
            "model_type": model_type,
            "target_column": target_column,
            "metrics": metrics,
        },
        "log": f"Successfully trained {model_type} Random Forest model on {output_view}. Model cached with ID: {model_id}"
    }


@mcp.tool("deploy_model_to_teradata")
def deploy_model_to_teradata_tool(
    model_id: str
) -> Dict[str, Any]:
    """
    Deploy a trained ONNX model to Teradata database using BYOM (Bring Your Own Model).
    
    Parameters
    ----------
    model_id : str
        Unique identifier of the cached trained model (from train_random_forest_model).
        
    Behavior
    --------
    - Retrieves the trained ONNX model from cache using model_id
    - Uploads model to Teradata using teradataml.save_byom() function
    - Creates View in Database using mldb.ONNXPredict which serves as model Endpoint.
    
    Important Notes
    ---------------
    - Once deployed, the model can be used with ONNXPredict functions in SQL
    
    Returns
    -------
    dict with:
      artifacts:
        - deployment_id: unique identifier for this deployment
        - model_id: original trained model ID from cache
        - teradata_model_id: identifier for the model in Teradata
        - model_description: description stored in Teradata
        - model_version: version stored in Teradata
        - model_type: classification or regression
        - target_column: name of the target variable
        - num_features: number of input features expected
        - deployment_status: "deployed"
      log: deployment status message with Teradata model ID
    """
    if not model_id:
        raise ValueError("model_id is required")
    
    create_db_context(HOST, USER, PASSWORD)
    try:
        view_name, view_sql = deploy_onnx_model_to_teradata(
            model_id=model_id
        )
    finally:
        close_db_context()

    return {
        "artifacts": {
            "model_id": model_id,
            "view_name": view_name,
            "view_ddl": view_sql,
        },
        "log": f"Successfully deployed ONNX model with model_id {model_id} to Teradata in model table models_table; accessible under view {view_name} "
    }




@mcp.tool("make_predictions")
def make_predictions(
    view_name:str
) -> Dict[str, Any]:
    """
    Test a model endpoint view by querying it and returning sample results.
    
    Parameters
    ----------
    view_name : str
        Name of view of model endpoint.

    Behavior
    --------
    - Generates and executes a test query against the endpoint view using SELECT * FROM view_name
    - Returns sample predictions with forwarded columns and json_report (contains predictions)
    
    Returns
    -------
    dict with:
      artifacts:
        - endpoint_view: fully-qualified endpoint view name
        - test_query: SQL query used for testing
        - row_count: number of rows returned
        - sample_results: first few prediction results
      log: test execution status message
    """
    if not view_name:
        raise ValueError("view_name is required")

    
    create_db_context(HOST, USER, PASSWORD)
    time.sleep(1)
    try:
        # Generate test SQL
        logging.info(f"Querying {view_name} for inference results")
        df_pred = tdml.DataFrame.from_query(f"SELECT * FROM {view_name}").to_pandas()
        logging.info(f"Querying {view_name} completed")
    finally:
        close_db_context()

    pred_records = df_pred.to_dict("records")

    
    return {
        "artifacts": {
            "pred_records": pred_records,
        },
        "log": f"Successfully tested endpoint view {view_name}. Retrieved {len(pred_records)} rows with predictions."
    }


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')