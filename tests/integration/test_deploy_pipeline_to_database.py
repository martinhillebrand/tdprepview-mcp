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

def test_deploy_pipeline_to_database_twice():
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
    finally:
        close_db_context()

    get_dummy_data_upload(dataset="boston_housing", schema=schema, table=table, if_exists="replace")

    # --- create pipeline ---
    res_pipe = create_ml_autoprep_pipeline(schema=schema, table=table, non_feature_cols=["row_id"])
    pipeline_id = res_pipe["artifacts"]["pipeline_id"]

    cache = get_cache()
    assert pipeline_id in cache.pipelines

    # --- first deployment ---
    res1 = deploy_pipeline_to_database(
        pipeline_id=pipeline_id,
        input_schema=schema,
        input_table=table,
        output_schema=schema,
        output_view_name=view_name,
    )
    # check cache for deployed view
    cache = get_cache()
    assert f"{schema}.{view_name}" in cache.prep_views
    entry = cache.prep_views[f"{schema}.{view_name}"]
    assert entry[0] == pipeline_id  # pipeline_id is first element

    assert res1["artifacts"]["pipeline_id"] == pipeline_id
    assert res1["artifacts"]["output_view"].lower() == f"{schema}.{view_name}".lower()
    assert res1["artifacts"]["column_count"] > 0

    DF_transformed = None
    try:
        create_db_context(host, user, pwd)
        try:
            DF_transformed = tdml.DataFrame(view_name)
            logging.info("DF transformed created with columns: %s", str(DF_transformed.columns))
        except Exception:
            pass
        assert DF_transformed

    finally:
        close_db_context()



    # --- second deployment (should replace view cleanly) ---
    res2 = deploy_pipeline_to_database(
        pipeline_id=pipeline_id,
        input_schema=schema,
        input_table=table,
        output_schema=schema,
        output_view_name=view_name,
    )

    # check cache for deployed view
    cache = get_cache()
    assert f"{schema}.{view_name}" in cache.prep_views
    entry = cache.prep_views[f"{schema}.{view_name}"]
    assert entry[0] == pipeline_id  # pipeline_id is first element

    assert res2["artifacts"]["pipeline_id"] == pipeline_id
    assert res2["artifacts"]["output_view"].lower() == f"{schema}.{view_name}".lower()
    assert res2["artifacts"]["column_count"] > 0

    logging.info("content of cache pipelines " + str(cache.prep_views))

    # --- cleanup: drop table + view ---
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
