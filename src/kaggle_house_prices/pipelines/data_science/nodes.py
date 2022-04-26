"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""
import logging

import pandas as pd
import numpy as np
from sklearn import model_selection

log = logging.getDebugger()

def train_test_split(*inputs, **kwargs):
    log.debug(f"{len(inputs)} input vectors")
    outputs = model_selection.train_test_split(*inputs, **kwargs)
    log.debug(f"{len(outputs)} output vectors")
    return outputs

def node_preprocess_df_for_model(df, preprocess_params=None, target_name=None):
    return preprocess_df_for_model(df, target_name=target_name, **preprocess_params)

def preprocess_df_for_model(
    df: pd.DataFrame,
    numeric_only: bool = True,
    drop_columns: list = None,
    keep_columns: list = None,
    target_name: str = None
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
        log.debug(f"Input df shape: {df.shape}")
    if target_name:
        X = df.drop(columns=target_name)
        y = df[target_name]
        log.debug(f"Returning tuple of X and y={target_name}")
        return (X, y)
    else:
        return df
    
def get_model_object(model_name):
    if model_name == "rforest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor()
    elif model_name == "elastic":
        from sklearn.linear_model import ElasticNet
        return ElasticNet()
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def node_train_model_on_Xy(X, y, model_object, model_options):
    model_init_params = model_options.get("model_init_params", {})
    model_fit_params = model_options.get("model_fit_params", {})
    model_instance = model_object(**model_init_params)
    model_instance.fit(X, y, **model_fit_params)

    return model_instance

