import os
import sys
import warnings
from contextlib import contextmanager

@contextmanager
def suppress_io_and_warnings():
    """Temporarily silence stdout, stderr, and warnings (use only around noisy imports/calls)."""
    orig_out, orig_err = sys.stdout, sys.stderr
    f = open(os.devnull, "w")
    sys.stdout = sys.stderr = f
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        f.close()


def create_db_context(host: str, user: str, password: str):
    """Create Teradata database connection context."""
    with suppress_io_and_warnings():
        import teradataml as tdml
        tdml.create_context(host, user, password)


def close_db_context():
    """Close Teradata database connection context."""
    with suppress_io_and_warnings():
        import teradataml as tdml
        tdml.remove_context()