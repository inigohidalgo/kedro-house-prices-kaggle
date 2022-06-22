"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""
import logging
import importlib
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

set_tracking_uri("http://127.0.0.1:12347/")


log = logging.getLogger(__name__)


def import_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.

    From: https://stackoverflow.com/a/34963527/9807171
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    try:
        return getattr(module_path, class_name)
    except AttributeError as err:
        raise ImportError('Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)) from err


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


class ClassHolder:
    def __init__(self, accept_module_str=True):
        self.classes = {}
        self.flexible_import = accept_module_str

    def add_class(self, c, class_name=None):
        key = class_name if class_name else c.__name__
        self.classes[key] = c

    def held(self, c):
        self.add_class(c)

        return c

    def __getitem__(self, key):

        try:
            return self.classes[key]
        except KeyError as e:
            if self.flexible_import:
                return import_string(key)
            else:
                raise e


model_holder = ClassHolder()
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

model_holder.add_class(RandomForestRegressor)
model_holder.add_class(ElasticNet)
