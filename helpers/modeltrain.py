"""Model training module for automated ML model training and ONNX conversion.

This module provides functionality for training Random Forest models on preprocessed
database views, converting models to ONNX format, and caching for deployment.
"""

import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple
import logging
from utils import suppress_io_and_warnings
from .cache import get_cache
import time

with suppress_io_and_warnings():
    import teradataml as tdml
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    from skl2onnx import to_onnx


def train_random_forest_model_payload(
    full_view_name: str,
    target_column: str,
    model_type: Literal["classification", "regression"],

) -> Tuple[str, Dict[str, Any]]:
    """Train a Random Forest model on a preprocessed database view.

    Uses column information from the pipeline cache to identify feature columns,
    trains a Random Forest model, converts to ONNX format, and caches the result.

    Args:
        view_schema: Schema name containing the preprocessed view
        view_name: Name of the preprocessed view to train on
        pipeline_id: Pipeline ID to retrieve column information from cache
        target_column: Name of the target variable column
        model_type: Type of model - "classification" or "regression"
        test_size: Fraction of data to use for testing. Defaults to 0.2.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple containing:
        - model_id: Unique UUID string identifier for the cached model
        - metrics: Dictionary containing model performance metrics



    """
    test_size: float = 0.2
    random_state: int = 42


    # Get column information from cache
    cache = get_cache()
    view_tuple = cache.prep_views.get(full_view_name)

    if view_tuple is None:
        raise ValueError(f"no view with given name {full_view_name}")

    (pipeline_id, input_schema, input_table, output_feat_colnames, output_nonfeat_colnames) = view_tuple

    logging.info(", ".join([str(x) for x in [pipeline_id, input_table, output_nonfeat_colnames, output_feat_colnames]]))


    logging.info("getting training data... " + str(full_view_name))
    df = tdml.DataFrame.from_query(f"SELECT * FROM {full_view_name}").to_pandas()
    logging.info("getting training data  done" + str(full_view_name))
    time.sleep(3)
    if df.empty:
        raise ValueError(f"No data retrieved from view {full_view_name}")
    
    # Prepare features and target
    X = df[output_feat_colnames].values
    y = df[target_column].values
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logging.info("model training started")
    # Train model
    if model_type == "classification":
        model = RandomForestClassifier(random_state=random_state)
    else:
        model = RandomForestRegressor(random_state=random_state)

    with suppress_io_and_warnings():
        model.fit(X_train, y_train)

    logging.info("model training done")
    
    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    
    if model_type == "classification":
        # Get classification report as dict
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'classification_report': class_report,
        }
    else:
        # Regression metrics
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        
        metrics = {
            'model_type': model_type,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'test_samples': len(y_test),
            'train_samples': len(y_train)
        }
    
    # Convert to ONNX
    num_features = len(output_feat_colnames)
    initial_types = [("features", FloatTensorType([None, num_features]))]

    # Generate unique model ID
    model_id = str(uuid.uuid4())
    logging.info("model conversion started")
    with suppress_io_and_warnings():
        model_onnx = to_onnx(
            model,
            name= model_id,
            initial_types = initial_types,
            target_opset=11
        )
    logging.info("model conversion done")
    

    
    # Cache the model and metadata
    model_metadata = {
        'model_id': model_id,
        'pipeline_id': pipeline_id,
        'full_view_name_training': full_view_name,
        'target_column': target_column,
        'feature_columns': output_feat_colnames,
        'non_feature_columns': output_nonfeat_colnames,
        'model_type': model_type,
        'num_features': num_features,
        'metrics': metrics,
        'random_state': random_state,
        'test_size': test_size
    }

    # save the model to file
    from .autoprep import EXCHANGE_DIR
    filename = f"{model_id}.onnx"
    EXCHANGE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = EXCHANGE_DIR / filename

    logging.info("starting writing to file " + str(file_path))

    with open(file_path, "wb") as f:
        f.write(model_onnx.SerializeToString())

    logging.info("writing to file done " + str(file_path))

    # Cache the model
    cache.models[model_id] = {
                        'onnx_model': model_onnx,
                        'onnx_path': file_path,
                        'metadata': model_metadata
                    }
    
    return model_id, metrics


