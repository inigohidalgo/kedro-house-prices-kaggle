"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""

import pandas as pd
import numpy as np
import logging
from kaggle_house_prices.utils import (
    date_utils as dt_utils,
    data_processing_utils as dp_utils,
)

log = logging.getLogger(__name__)


def raw_to_intermediate_houses(houses_dataset):
    houses_dataset["CentralAir"] = dp_utils._is_true(houses_dataset["CentralAir"])

    houses_dataset["DtSold"] = dt_utils.get_date_from_columns(
        houses_dataset, {"year": "YrSold", "month": "MoSold"}
    )
    return houses_dataset


def preprocess_df_for_model(df, model_params=None):
    # TODO: try to extract config to separate yaml file
    preprocess_params = model_params.get("preprocess_params")
    if preprocess_params is None:
        preprocess_params = {}
    if "numeric_only" not in preprocess_params:
        log.info("numeric_only not in preprocess_params. Setting to True.")
        preprocess_params["numeric_only"] = True
    return subset_columns(df, **preprocess_params)


# todo: node to rename columns, either by parameters in yaml or camel case to snake case


def subset_columns(
    df: pd.DataFrame,
    numeric_only: bool = False,
    drop_columns: list = None,
    keep_columns: list = None,
) -> pd.DataFrame:

    log.debug(f"Input df shape: {df.shape}")
    cols_in = df.columns
    if numeric_only:
        log.debug("Dropping non-numeric columns")
        df = df.select_dtypes(np.number)
    if drop_columns:
        log.debug(f"Dropping columns: {drop_columns}")
        df = df.drop(columns=drop_columns)
    if keep_columns:
        log.debug(f"Dropping all columns except: {keep_columns}")
        df = df[keep_columns]
    cols_out = df.columns
    cols_missing = set(cols_in) - set(cols_out)
    if cols_missing:
        log.info(f"Columns removed from df before training: {cols_missing}")
        log.debug(f"Output df shape: {df.shape}")
    return df


def clean_houses(houses_dataset):
    return houses_dataset
