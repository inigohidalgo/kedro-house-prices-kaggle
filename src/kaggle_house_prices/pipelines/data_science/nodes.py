"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""
import logging

import pandas as pd
import numpy as np
from sklearn import model_selection

from kaggle_house_prices.utils import data_processing_utils as dp_utils

log = logging.getLogger(__name__)


def train_test_split(input_df, train_test_split_options):
    # log.debug(f"{len(inputs)} input vectors")
    outputs = model_selection.train_test_split(input_df, **train_test_split_options)
    # log.debug(f"{len(outputs)} output vectors")
    return outputs


def train_model_on_df(train_df, model_class, model_options,):
    model_init_params = model_options.get("model_init_params", {})
    model_fit_params = model_options.get("model_fit_params", {})
    model_object = model_class(**model_init_params)
    X, y = dp_utils.extract_target_from_df(train_df, model_options.get("target_name"))
    model_object.fit(X, y, **model_fit_params)

    return model_object


def get_model_class(model_name):
    """
    Returns the model class
    """
    if model_name == "rforest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor
    elif model_name == "elastic":
        from sklearn.linear_model import ElasticNet
        return ElasticNet
    else:
        raise ValueError(f"Unknown model name: {model_name}")
