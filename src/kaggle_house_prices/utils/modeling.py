from __future__ import annotations

from typing import Literal, Optional
from sklearn import ensemble as sk_ensemble, linear_model as sk_lm  # type: ignore
import pandas as pd


from typing import Mapping


def get_model_class(model_type: Literal["regression", "classification"], model_name: str) -> type[AbstractGenericModel]:
    regression_models = {
        "rforest": sk_ensemble.RandomForestRegressor,
        "elasticnet": sk_lm.ElasticNet,
    }

    classification_models = {
        "rforest": sk_ensemble.RandomForestClassifier,
        # "logistic": sk_lm.LogisticClassifier,
    }

    models = {
        "regression": regression_models,
        "classification": classification_models,
    }

    return models[model_type][model_name]


from typing import runtime_checkable, Protocol, Any, Union, overload, TypeVar
import numpy as np

Table = TypeVar("Table", np.ndarray, pd.DataFrame)


@runtime_checkable
class Predictor(Protocol):
    # @overload
    def predict(self, X):
        ...


class Fitter(Protocol):
    @overload
    def fit(self, X, y):
        ...

    @overload
    def fit(self, X):
        ...


@runtime_checkable
class AbstractGenericModel(Fitter, Predictor, Protocol):
    # def __init__(self, model_class, model_init_params: Optional[Mapping] = None):
    #     self.model_class = model_class
    #     if not model_init_params:
    #         model_init_params = {}
    #     self.model_init_params = model_init_params
    ...
    # def fit(self, X, y, **kwargs):
    #     ...

    # def predict(self, X, **kwargs):
    #     ...

from typing import Generic

class GenericModel(AbstractGenericModel):
    def __init__(self, model_class):
        self.model_class = model_class

    def predict(self, X: Table, **kwargs) -> Table:
        return self.model_class.predict(X, **kwargs)


@runtime_checkable
class TargetGenericModel(AbstractGenericModel, Protocol):
    target_name: str

    def fit(self, X):
        ...

from numbers import Number


class Key(Number, str):
    ...

# KeyT = TypeVar("KeyT", bound=Key, contravariant=True)

class BaseCombinedModel(AbstractGenericModel):
    """Generic model that can make multiple predictions"""
    # target_model_classes: Optional[Mapping[KeyT, type[GenericModel]]]

    def __init__(
        self,
        model_classes: Optional[Mapping[Key, type[GenericModel]]] = None,
        model_init_params: Optional[Mapping] = None,
    ):
        if not model_classes:
            self.target_model_classes = model_classes

        if not model_init_params:
            model_init_params = {}
        self.model_init_params = model_init_params
        self.models_instantiated = False

    def fit(self, X, **kwargs):
        ...
        # for model_object in self.target_model_objects.values():
        #     model_object.fit(X, **kwargs)

    def predict(self, X: Table):
        ...
        # try:
        #     predictions = pd.DataFrame(index=X.index)
        # except AttributeError:
        #     predictions = np.empty(X.shape[0])
        # for target_name, model_object in self.target_model_objects.items():
        #     predictions[target_name] = model_object.predict(X)
        # return self.model_class.predict(X)

    def instantiate_model_objects(self):
        self.model_objects = {}
        for target_name, model_class in self.target_model_classes.items():
            self.target_model_objects[target_name] = model_class(
                **self.model_init_params[target_name],
            )


from typing import Iterable


class QuantileModel(BaseCombinedModel):
    def __init__(self, model_class: type[GenericModel], quantiles : Iterable[Number], **kwargs):
        model_classes = {quantile: model_class for quantile in quantiles}
        super().__init__(model_classes=model_classes, **kwargs)
        self.quantiles = quantiles
