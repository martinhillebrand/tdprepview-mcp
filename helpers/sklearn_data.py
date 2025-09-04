from typing import Literal
import pandas as pd
from utils import suppress_io_and_warnings
import logging

with suppress_io_and_warnings():
    import teradataml as tdml
    from sklearn.datasets import (
        load_iris,
        load_diabetes,
        load_linnerud,
        load_wine,
        load_breast_cancer,
        fetch_california_housing,
        fetch_openml
    )

import re

def _make_db_friendly(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    if not re.match(r'^[a-z]', name):
        name = "col_" + name
    return name.strip('_')

def load_sklearn_df(dataset: Literal["iris", "diabetes", "linnerud", "wine", "breast_cancer", "california_housing", "boston_housing", "titanic", "adult_census"]) -> pd.DataFrame:
    if dataset in ["iris", "diabetes", "linnerud", "wine", "breast_cancer"]:
        loaders = {
            "iris": load_iris,
            "diabetes": load_diabetes,
            "linnerud": load_linnerud,
            "wine": load_wine,
            "breast_cancer": load_breast_cancer,
        }
        df = loaders[dataset](as_frame=True).frame
    elif dataset == "california_housing":
        df = fetch_california_housing(as_frame=True).frame
    elif dataset == "boston_housing":
        data = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
        df = pd.concat([data.data, data.target], axis=1)
    elif dataset == "titanic":
        data = fetch_openml(name="titanic", version=1, as_frame=True, parser="auto")
        df = pd.concat([data.data, data.target], axis=1)
    elif dataset == "adult_census":
        data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
        df = pd.concat([data.data, data.target], axis=1)
    else:
        available = ["iris", "diabetes", "linnerud", "wine", "breast_cancer", "california_housing", "boston_housing", "titanic", "adult_census"]
        raise ValueError(f"Unknown dataset: {dataset}. Available: {available}")

    df = df.reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))
    df.columns = [_make_db_friendly(c) for c in df.columns]

    return df

def upload_df_to_teradata(df: pd.DataFrame,
                         schema: str,
                         table: str,
                         if_exists: Literal["fail", "replace", "append"]) -> None:
    """
    Upload a pandas DataFrame to Teradata using teradataml.copy_to_sql
    """
    logging.info("uploading pandas DataFrame to Teradata " + table)
    tdml.copy_to_sql(
        df=df,
        table_name=table,
        schema_name=schema,
        if_exists=if_exists,
        index=False,
        primary_index="row_id"
    )
    logging.info("uploading pandas DataFrame to Teradata completed ")