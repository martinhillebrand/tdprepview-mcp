# tests/integration/test_get_dummy_data_upload.py

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

def test_get_dummy_data_upload_roundtrip():
    host = _need("DB_HOST")
    user = _need("DB_USER")
    pwd  = _need("DB_PASSWORD")
    schema = os.getenv("DB_SCHEMA", user)
    table = f"mcp_{uuid.uuid4().hex[:8]}"
    fq = f"{schema}.{table}"

    from server import get_dummy_data_upload, create_db_context, close_db_context
    import teradataml as tdml

    # --- pre-drop if exists ---
    create_db_context(host, user, pwd)
    try:
        try:
            tdml.db_drop_table(fq)  # adjust if needed
        except Exception:
            pass
    finally:
        close_db_context()

    # --- call function under test (manages its own context) ---
    res = get_dummy_data_upload(dataset="iris", schema=schema, table=table, if_exists="replace")
    assert isinstance(res, dict)
    art = res["artifacts"]
    assert art["input_table"].lower() == fq.lower()
    assert art["dataset"] == "iris"
    assert art["rows"] > 0
    assert "row_id" in art["columns"]

    # --- verify table exists ---
    create_db_context(host, user, pwd)
    try:
        DF_upload = tdml.DataFrame(tdml.in_schema(schema, table))
        a, b = DF_upload.shape
        collist = DF_upload.columns

        # assertions
        assert isinstance(a, int) and a >= 0
        assert isinstance(b, int) and b >= 0
        assert isinstance(collist, list)
        assert all(isinstance(c, str) for c in collist)

        # log useful info for debugging
        logging.info("Uploaded table shape: (%d, %d)", a, b)
        logging.info("Uploaded table columns: %s", collist)

        tables = tdml.db_list_tables(schema).TableName.values
        is_table_in_db = table in tables
        assert is_table_in_db, f"{fq} not found in db_list_tables({schema})"
    finally:
        # cleanup
        try:
            tdml.db_drop_table(fq)
        except Exception:
            pass
        close_db_context()
