"""Model deployment module for uploading ONNX models to Teradata using BYOM.

This module provides functionality for deploying trained ONNX models from cache
to Teradata database using the Bring Your Own Model (BYOM) functionality.
"""

import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time
import logging
from .cache import get_cache
from utils import suppress_io_and_warnings

with suppress_io_and_warnings():
    import teradataml as tdml



def deploy_onnx_model_to_teradata(
    model_id: str,
) -> Tuple[str, str]:
    """Deploy a cached ONNX model to Teradata database using BYOM save_byom function.
    
    Takes a model from the training cache, temporarily saves it as an ONNX file,
    and uploads it to Teradata using the BYOM (Bring Your Own Model) functionality.
    
    Args:
        model_id: Unique UUID string identifier for the cached trained model

    """
    # Get column information from cache
    cache = get_cache()
    models_tuple = cache.models.get(model_id)

    if models_tuple is None:
        raise ValueError(f"no model with given id {model_id}")

    #upload model
    onnx_path = models_tuple.get("onnx_path")
    with suppress_io_and_warnings():
        try:
            logging.info("trying to delete models_table")
            tdml.execute_sql(f"DROP TABLE models_table")
            logging.info("succeeded")
        except:
            logging.info("failed because not existing")
            pass
        time.sleep(2)
        logging.info("uploading ONNX file...")
        tdml.save_byom(model_id, model_file=str(onnx_path), table_name="models_table")
        logging.info("uploading ONNX file done")

    # create view = deployment
    """
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
    """
    model_metadata = models_tuple.get("metadata")
    full_view_name_training = model_metadata.get("full_view_name_training")
    non_feature_columns = model_metadata.get("non_feature_columns")
    feature_columns = model_metadata.get("feature_columns")
    target_column = model_metadata.get("target_column")

    accumulate_cols = ",".join([c for c in non_feature_columns if c != target_column])
    my_model_input_fields_map = "features="+ ",".join(feature_columns)

    model_id_db = model_id.replace("-", "")[:12]
    view_name = f"predict_model_{model_id_db}"

    try:
        logging.info("dropping inference view... ")
        tdml.execute_sql(f"DROP VIEW {view_name}")
        logging.info("done ")
    except:
        logging.info("did not exist ")
        pass

    time.sleep(2)

    view_sql = f"""
REPLACE VIEW {view_name} AS
SELECT  
    *
FROM mldb.ONNXPredict ( 
    ON (SELECT TOP 10 * FROM {full_view_name_training} ) AS InputTable 
    PARTITION BY ANY  
    ON ( SELECT model_id, model FROM models_table WHERE model_id = '{model_id}'
    ) AS ModelTable 
    DIMENSION   
    USING
    Accumulate('{accumulate_cols}')
    ModelInputFieldsMap('{my_model_input_fields_map}')
) AS sqlmr   

"""
    logging.info("view ddl " + view_sql)
    logging.info("replacing view ... " + view_name)
    tdml.execute_sql(view_sql)
    logging.info("replacing view done " + view_name)

    cache.deployments[model_id] = view_name

    return view_name, view_sql




