"""AutoPrep module for automated pipeline creation and management.

This module provides functionality for creating, caching, and managing
tdprepview pipelines from Teradata tables, including Sankey diagram generation
and file exchange capabilities.
"""
import logging

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from utils import suppress_io_and_warnings
from .cache import get_cache

with suppress_io_and_warnings():
    import teradataml as tdml
    import tdprepview

# Configuration for file exchange
SERVER_DIR = Path(__file__).resolve().parent
EXCHANGE_DIR = Path(
    (Path.environ.get("MCP_EXCHANGE_DIR") if hasattr(Path, "environ") else None)
    or SERVER_DIR / "exchange"
).resolve()


def create_and_fit_pipeline(
    schema: str,
    table: str,
    non_feature_cols: Optional[List[str]] = None
) -> Tuple[str, str]:
    """Create and fit a tdprepview Pipeline from a Teradata table.
    
    This function automatically generates pipeline steps using tdprepview.auto_code,
    creates a Pipeline object, fits it, and caches the result for later use.

    """
    if non_feature_cols is None:
        non_feature_cols = []

    pipeline_steps_str = tdprepview.auto_code(
        DF= None,
        input_schema=schema,
        input_table=table,
        non_feature_cols=non_feature_cols
    )

    pipeline = tdprepview.Pipeline(steps=eval(pipeline_steps_str))
    logging.info("fitting started")
    with suppress_io_and_warnings():
        pipeline.fit(schema_name=schema, table_name=table)
    logging.info("fitting done")
    # Generate unique ID for caching
    pipeline_id = str(uuid.uuid4())

    pipeline_dict = pipeline.to_dict()
    list_of_feature_output_column_names = [c for c in pipeline.get_output_column_names() if c not in non_feature_cols]

    # Cache the fitted pipeline
    cache = get_cache()
    cache.pipelines[pipeline_id] =  ( list_of_feature_output_column_names,pipeline_dict)
    
    return pipeline_id, pipeline_steps_str



def generate_and_save_plotly_figure(
        pipeline_id: str,
        filename: Optional[str] = None,
        height: int = 800,
        width: int = 1200,
        ) -> Path:
    """Save a plotly figure to the exchange directory and return absolute path.

    """
    cache = get_cache()
    pl_tuple =  cache.pipelines.get(pipeline_id)
    if pl_tuple is None:
        raise ValueError(f"no pipeline with given id {pipeline_id}")

    (_, pipeline_dict) = pl_tuple
    pl = tdprepview.Pipeline.from_dict(pipeline_dict)

    fig = pl.plot_sankey()
    fig.update_layout(height=height, width=width)

    file_name = filename or f"pipeline_{pipeline_id}"
    fname = f"{file_name}.html"
    EXCHANGE_DIR.mkdir(parents=True, exist_ok=True)

    file_path = (EXCHANGE_DIR / fname).resolve()
    fig.write_html(file_path, full_html=True, include_plotlyjs="inline")

    return file_path




def deploy_pipeline_as_view(
    pipeline_id: str,
    input_schema: str,
    input_table: str,
    output_schema: str,
    output_view_name: str = None
) -> Tuple[str, Any]:
    """Deploy a cached pipeline as a database view using the transform method.
    
    Creates a database view that applies the pipeline transformations to the input table.
    This acts as a 'deploy button' that crystallizes the preprocessing logic into the database.
    
    Args:
        pipeline_id: Unique UUID string identifier for the cached pipeline
        input_schema: Schema name of the source table
        input_table: Table name to transform
        output_schema: Schema where the view will be created
        output_view_name: Name for the output view.
    
    Returns:
        - view_name: Fully qualified name of the created view (schema.view)
        - output_columns_names: list of column names for the output view
    """
    cache = get_cache()

    pl_tuple = cache.pipelines.get(pipeline_id)
    if pl_tuple is None:
        raise ValueError(f"no pipeline with given id {pipeline_id}")

    logging.info("creating pipelein from dict .... started")
    (list_of_feature_output_column_names, pipeline_dict) = pl_tuple
    pipeline = tdprepview.Pipeline.from_dict(pipeline_dict)
    logging.info("creating pipelein from dict .... done")




    # Deploy pipeline as view using transform method
    with suppress_io_and_warnings():
        try:
            logging.info(f"dropping view  {output_schema}.{output_view_name} ")
            tdml.execute_sql(f"DROP VIEW {output_schema}.{output_view_name}")
            logging.info(f"dropping view  {output_schema}.{output_view_name} done ")
        except:
            pass

        time.sleep(3)

        logging.info(f"pipeline transform   {output_view_name} ... ")
        DF_tr = pipeline.transform(
            schema_name=input_schema,
            table_name=input_table,
            create_replace_view=True,
            return_type = "df",
            output_schema_name=output_schema,
            output_view_name=output_view_name
        )
        logging.info(f"pipeline transform   {output_view_name} done")

    output_columns_names = DF_tr.columns

    output_feat_colnames = [c for c in output_columns_names if c in list_of_feature_output_column_names]
    output_nonfeat_colnames = [c for c in output_columns_names if c not in output_feat_colnames]

    # Return fully qualified view name
    full_view_name = f"{output_schema}.{output_view_name}"

    cache.prep_views[full_view_name] = (pipeline_id, input_schema, input_table, output_feat_colnames, output_nonfeat_colnames)
    
    return full_view_name, output_columns_names
