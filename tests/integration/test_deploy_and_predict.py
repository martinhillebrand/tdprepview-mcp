import os
import uuid
import pytest
import logging
pytestmark = pytest.mark.integration

def _need(var):
    val = os.getenv(var)
    if not val:
        pytest.skip(f"Missing {var} in env")
    return val

def test_deploy_and_make_predictions():
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
        deploy_model_to_teradata_tool,
        make_predictions,
        create_db_context,
        close_db_context,
    )
    import teradataml as tdml
    from helpers.cache import get_cache

    # --- cleanup upfront ---
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

    # --- upload + pipeline + view ---
    get_dummy_data_upload(dataset="boston_housing", schema=schema, table=table, if_exists="replace")
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

    # --- train model ---
    res_train = train_random_forest_model_tool(
        output_view=full_view,
        target_column="medv",
        model_type="regression",
    )
    model_id = res_train["artifacts"]["model_id"]

    cache = get_cache()
    assert model_id in cache.models

    # --- deploy model ---
    res_deploy = deploy_model_to_teradata_tool(model_id=model_id)
    art_dep = res_deploy["artifacts"]
    assert art_dep["model_id"] == model_id
    assert "view_name" in art_dep
    endpoint_view = art_dep["view_name"]

    assert model_id in cache.deployments
    assert cache.deployments[model_id] == endpoint_view

    # --- test predictions ---
    res_pred = make_predictions(view_name=endpoint_view)
    art_pred = res_pred["artifacts"]
    assert "pred_records" in art_pred
    assert isinstance(art_pred["pred_records"], list)
    assert len(art_pred["pred_records"]) > 0
    logging.info(str(art_pred["pred_records"]))
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
        try:
            tdml.execute_sql(f"DROP VIEW {endpoint_view}")
        except Exception:
            pass
        try:
            tdml.execute_sql("DROP TABLE models_table")
        except Exception:
            pass
    finally:
        close_db_context()
