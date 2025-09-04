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

def test_create_ml_autoprep_pipeline_roundtrip():
    host = _need("DB_HOST")
    user = _need("DB_USER")
    pwd  = _need("DB_PASSWORD")
    schema = os.getenv("DB_SCHEMA", user)
    table = f"mcp_{uuid.uuid4().hex[:8]}"
    fq = f"{schema}.{table}"

    from server import (
        get_dummy_data_upload,
        create_ml_autoprep_pipeline,
        create_db_context,
        close_db_context,
    )
    from helpers.cache import get_cache

    import teradataml as tdml

    # --- prepare: upload dummy dataset ---
    try:
        create_db_context(host, user, pwd)
        try:
            tdml.db_drop_table(fq)
        except Exception:
            pass
    finally:
        close_db_context()

    get_dummy_data_upload(dataset="boston_housing", schema=schema, table=table, if_exists="replace")

    # --- run pipeline creation ---
    res = create_ml_autoprep_pipeline(schema=schema, table=table, non_feature_cols=["row_id"])
    assert isinstance(res, dict)
    art = res["artifacts"]

    # basic checks
    assert art["input_table"].lower() == fq.lower()
    assert isinstance(art["pipeline_id"], str) and len(art["pipeline_id"]) > 0
    assert art["non_feature_cols"] == ["row_id"]
    assert art["pipeline_steps"]
    assert "Created and fitted ML preprocessing pipeline" in res["log"]

    logging.info("Pipeline steps: %s", art["pipeline_steps"])

    #check cache
    cache = get_cache()
    assert art["pipeline_id"] in cache.pipelines
    features, pipeline_dict = cache.pipelines[art["pipeline_id"]]
    assert isinstance(features, list)
    assert isinstance(pipeline_dict, dict)


    # --- cleanup table ---
    create_db_context(host, user, pwd)
    try:
        tdml.db_drop_table(fq)
    except Exception:
        pass
    finally:
        close_db_context()
