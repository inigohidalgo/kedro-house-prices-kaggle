"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""
import logging
from sklearn import model_selection

from mlflow import (
    log_metric,
    log_param,
    set_tracking_uri,
)

from kaggle_house_prices.utils import (
    data_processing as dp_utils,
    performance as performance_utils,
)

from kaggle_house_prices.utils.model.model_register import model_holder

set_tracking_uri("http://127.0.0.1:12347/")


log = logging.getLogger(__name__)


def load_model(model_name):
    return model_holder[model_name]

def train_test_split(input_df, train_test_split_options):
    # log.debug(f"{len(inputs)} input vectors")
    outputs = model_selection.train_test_split(input_df, **train_test_split_options)
    # log.debug(f"{len(outputs)} output vectors")
    return outputs


def train_model_on_df(
    train_df,
    model_class,
    model_options,
):
    model_init_params = model_options.get("model_init_params", {})
    log_param("model_init_params", model_init_params)
    model_fit_params = model_options.get("model_fit_params", {})
    model_object = model_class(**model_init_params)
    X, y = dp_utils.extract_target_from_df(train_df, model_options.get("target_name"))
    log.info("Fitting model")
    model_object.fit(X, y, **model_fit_params)
    scoring_function_name = model_options.get("scoring_function")
    if scoring_function_name:
        scoring_function = performance_utils.get_scoring_function(scoring_function_name)
        score = scoring_function(y, model_object.predict(X))
        log_metric(scoring_function_name, score)

    return model_object