import logging
from numbers import Number
import abc
from typing import (
    Any,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
    Type,
    Union,
    overload,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from sklearn import ensemble as sk_ensemble  # type: ignore
from sklearn import linear_model as sk_lm

logger = logging.getLogger()





# Table = TypeVar("Table", np.ndarray, pd.DataFrame)
Table = Union[np.ndarray, pd.DataFrame]


@runtime_checkable
class Predictor(Protocol):
    @abc.abstractmethod
    def predict(self, X, **kwargs):
        ...


class Fitter(Protocol):
    @abc.abstractmethod
    @overload
    def fit(self, X, y, **kwargs):
        ...

    @overload
    @abc.abstractmethod
    def fit(self, X, **kwargs):
        ...


@runtime_checkable
class AbstractGenericModel(Fitter, Predictor, Protocol):
    ...


@runtime_checkable
class TargetGenericModel(AbstractGenericModel, Protocol):
    target_name: str
    
    @abc.abstractmethod
    def fit(self, data, **kwargs):
        ...


class GenericModel(AbstractGenericModel):
    model_class: AbstractGenericModel

    def __init__(self, model_class):
        self.model_class = model_class

    def predict(self, X: Table, **kwargs) -> Table:
        return self.model_class.predict(X, **kwargs)


class TargetModel(TargetGenericModel, GenericModel):
    def __init__(self, model_class, target_name):
        GenericModel.__init__(model_class)
        self.target_name = target_name

    def fit(self, data: pd.DataFrame, **kwargs):
        X, y = self.get_Xy_from_data(data)
        GenericModel.fit(X, y, **kwargs)

    def get_Xy_from_data(self, data: pd.DataFrame):
        X = data.drop(self.target_name, axis=1)
        y = data[self.target_name]
        return X, y


KeyTs = Union[Number, str]
KeyT = TypeVar("KeyT", bound=KeyTs)
ModelObjMap = Mapping[KeyT, GenericModel]
ModelClsMap = Mapping[KeyT, Type[GenericModel]]


class BaseCombinedModel(AbstractGenericModel, Generic[KeyT]):
    """Generic model that can make multiple predictions"""

    _models_instantiated: bool

    # target_model_classes: Optional[Mapping[KeyT, Type[GenericModel]]]

    def __init__(
        self,
        model_classes: Optional[ModelClsMap] = None,
        model_init_params: Optional[Mapping] = None,
        model_objects: Optional[ModelObjMap] = None,
    ):
        if model_classes and not model_objects:
            self.target_model_classes: ModelClsMap = model_classes
            self._models_instantiated = False
        elif model_objects:
            self.target_model_objects: ModelObjMap = model_objects
            self._models_instantiated = True
        else:
            raise NotImplementedError(
                f"Must instantiate f{type(self).__name__} with either model_classes or model_objects"
            )

        if not model_init_params:
            model_init_params = {}
        self.model_init_params = model_init_params

    def fit(self, X, model_init_kwargs=None, **kwargs):

        self.instantiate_model_objects()
        # for model_object in self.target_model_objects.values():
        #     model_object.fit(X, **kwargs)

    # @abc.abstractmethod
    def predict(self, X: Table, **kwargs) -> pd.DataFrame:
        predictions = pd.DataFrame()
        # TODO: explicit index handling
        # try:
        #     predictions.index = X.index
        # except AttributeError:
        #     predictions.index = pd.RangeIndex(X.shape[0])
        for target_name, model_object in self.target_model_objects.items():
            predictions[target_name] = model_object.predict(X, **kwargs)
        return predictions

    def instantiate_model_objects(self, **additional_model_init_kwargs):
        if not self._models_instantiated:
            self.model_objects = {}
            for target_name, model_class in self.target_model_classes.items():
                model_init_kwargs = {**self.model_init_params[target_name], **additional_model_init_kwargs}
                self.target_model_objects[target_name] = model_class(
                    **model_init_kwargs,
                )
        else:
            logger.warning("Model objects already instantiated")


class QuantileModel(BaseCombinedModel[Number]):
    def __init__(self, model_class: Type[GenericModel], quantiles: Iterable[Number], **kwargs):
        model_classes = {quantile: model_class for quantile in quantiles}
        super().__init__(model_classes=model_classes, **kwargs)
        self.quantiles = quantiles


# TODO SMELL: should be able to abstract this out of this file. decouple
def get_model_class(model_type: Literal["regression", "classification"], model_name: str) -> Type[AbstractGenericModel]:
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