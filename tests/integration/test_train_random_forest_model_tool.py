import os
import uuid
import pytest

pytestmark = pytest.mark.integration

def _need(var):
    val = os.getenv(var)
    if not val:
        pytest.skip(f"Missing {var} in env")
    return val

def test_train_random_forest_model_tool_twice():
    host = _need("DB_HOST")
    user = _need("DB_USER")
    pwd  = _need("DB_PASSWORD")
    schema = os.getenv("DB_SCHEMA", user)
    table = f"mcp_{uuid.uuid4().hex[:8]}"
    view_name = f"{table}_prepared"
    fq = f"{schema}.{table}"

    from server import (
        get_dummy_data_upload,
        create_ml_autoprep_pipeline,
        deploy_pipeline_to_database,
        train_random_forest_model_tool,
        create_db_context,
        close_db_context,
    )
    import teradataml as tdml
    from helpers.cache import get_cache

    # --- prepare input table ---
    try:
        create_db_context(host, user, pwd)
        try:
            tdml.db_drop_table(fq)
        except Exception:
            pass
        try:
            tdml.execute_sql(f"DROP VIEW {schema}.{view_name}")
        except Exception:
            pass
    finally:
        close_db_context()

    get_dummy_data_upload(dataset="boston_housing", schema=schema, table=table, if_exists="replace")

    # --- pipeline + deployment ---
    res_pipe = create_ml_autoprep_pipeline(schema=schema, table=table, non_feature_cols=["row_id"])
    pipeline_id = res_pipe["artifacts"]["pipeline_id"]

    deploy_pipeline_to_database(
        pipeline_id=pipeline_id,
        input_schema=schema,
        input_table=table,
        output_schema=schema,
        output_view_name=view_name,
    )
    full_view = f"{schema}.{view_name}"

    # --- first training run ---
    res1 = train_random_forest_model_tool(
        output_view=full_view,
        target_column="medv",
        model_type="regression",
    )
    art1 = res1["artifacts"]
    assert isinstance(art1["model_id"], str)
    assert art1["model_type"] == "regression"
    assert art1["target_column"] == "medv"

    cache = get_cache()
    assert art1["model_id"] in cache.models
    onnx_path = cache.models[art1["model_id"]]["onnx_path"]
    assert onnx_path.exists() and onnx_path.is_file()


    # --- second training run (should produce a different model_id but succeed) ---
    res2 = train_random_forest_model_tool(
        output_view=full_view,
        target_column="medv",
        model_type="regression",
    )
    art2 = res2["artifacts"]
    assert art2["model_id"] != art1["model_id"]
    assert art2["model_id"] in cache.models
    onnx_path2 = cache.models[art2["model_id"]]["onnx_path"]
    assert onnx_path2.exists() and onnx_path2.is_file()

    # --- cleanup ---
    create_db_context(host, user, pwd)
    try:
        try:
            tdml.db_drop_table(fq)
        except Exception:
            pass
        try:
            tdml.execute_sql(f"DROP VIEW {schema}.{view_name}")
        except Exception:
            pass
    finally:
        close_db_context()
