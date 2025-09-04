import os
import uuid
import pytest
from dotenv import load_dotenv


# ensure `python-dotenv` in your uv project
load_dotenv(override=False)

def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        pytest.skip(f"Missing {var} in env; skipping integration test.")
    return val

@pytest.fixture(scope="session")
def db_env():
    host = _require_env("DB_HOST")
    user = _require_env("DB_USER")
    pwd  = _require_env("DB_PASSWORD")
    schema = os.getenv("DB_SCHEMA", user)
    return {"host": host, "user": user, "pwd": pwd, "schema": schema}

@pytest.fixture
def unique_table():
    return f"mcp_{uuid.uuid4().hex[:8]}"

# Thin wrappers so test uses the same helpers you use in server.py
@pytest.fixture
def server_helpers():
    # server.py must export these (or adjust import path)
    from server import create_db_context, close_db_context
    import teradataml as tdml
    return {"create_ctx": create_db_context, "close_ctx": close_db_context, "tdml": tdml}

@pytest.fixture
def db_context(db_env, server_helpers):
    sh = server_helpers
    sh["create_ctx"](db_env["host"], db_env["user"], db_env["pwd"])
    try:
        yield
    finally:
        sh["close_ctx"]()

def _safe_drop(tdml, fqname: str):
    try:
        tdml.db_drop_table(fqname)  # user’s requested API
    except Exception:
        pass  # ignore if missing
