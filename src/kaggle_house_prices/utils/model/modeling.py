import logging
from numbers import Number
import abc
from typing import (
    Any,
    Dict,
    Generic,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
    List,
    Type,
    Union,
    overload,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from sklearn import ensemble as sk_ensemble  # type: ignore
from sklearn import linear_model as sk_lm
from numpy.typing import ArrayLike
from typing import MutableMapping

from kaggle_house_prices.utils.data.data_processing import (
    extract_target_from_df,
)

from kaggle_house_prices.utils.model.typing import AbstractModel, ModelT, ModelCls

from kaggle_house_prices.utils.data.typing import DataHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# DataHandler = Callable[[DataT, str, Optional[Mapping]], DataT]


class Model(Generic[ModelT]):
    """
    A model wrapper implementing unified interface.
    """

    def __init__(
        self,
        model_class: Type[ModelT] = None,
        model_object: ModelT = None,
        instantiate: bool = False,
        data_handlers: Optional[Iterable[DataHandler]] = None,
        **model_params: Dict[str, Any],
    ) -> None:

        if model_class is not None:
            self.model_class = model_class
            self.__instantiated = False
        elif model_object is not None:
            self.model_object = model_object
            self.__instantiated = True
        else:
            raise ValueError("model_class or model_object must be set")

        if "model_params" in model_params:
            self.model_params = model_params["model_params"]
        else:
            self.model_params = model_params

        if data_handlers is None:
            data_handlers = []
        self.data_handlers = data_handlers

        if instantiate:
            self.instantiate()

    def instantiate(self, **model_init_kwargs: Dict[str, Any]) -> None:
        self.set_params(**model_init_kwargs)

        if self.__instantiated:
            logger.info(f"{type(self).__name__}")
        else:
            logger.debug(f"Instantiating model with {self.model_params}")
            self.model_object = self.model_class(**self.model_params)
            self.__instantiated = True

    def set_params(self, **kwargs: ModelCfg) -> None:
        self.model_params.update(kwargs)
        if self.__instantiated:
            self.model_object.set_params(**kwargs)

    def handle_data(self, data: DataT, mode: DataHandlingModeT, **kwargs: Dict[str, Any]) -> DataT:

        for data_handler in self.data_handlers:
            data = data_handler(data, mode, **kwargs)
        return data

    def fit(self, data: DataT, **train_kwargs: Dict[str, Any]) -> None:
        if not self.__instantiated:
            logger.debug(f"Instantiating {self}")
            self.instantiate()

        data = self.handle_data(data, "train")

        if not train_kwargs:
            train_kwargs = {}

        if isinstance(data, dict):
            self.model_object.fit(**data, **train_kwargs)
        elif isinstance(data, tuple):
            self.model_object.fit(*data, **train_kwargs)
        else:
            self.model_object.fit(data, **train_kwargs)

    def __repr__(self) -> str:
        if self.__instantiated:
            return f"{self.model_object}"
        else:
            return f"{self.model_class}"


ModelObjMap = Mapping[KeyT, ModelT]
ModelClsMap = Mapping[KeyT, ModelCls]


class MultiModel(MutableMapping):
    def __init__(
        self,
        model_classes: Optional[ModelClsMap] = None,
        model_params: Optional[ModelCfgMapping] = None,
        model_objects: Optional[ModelObjMap] = None,
        instantiate: bool = True,
    ):
        if model_classes and model_objects:
            raise ValueError("model_classes and model_objects cannot both be set")

        if model_classes:
            self.model_classes: ModelClsMap = model_classes
            self.__instantiated = False
        elif model_objects:
            self.model_objects: ModelObjMap = model_objects
            self.__instantiated = True
        else:
            raise NotImplementedError(
                f"Must instantiate f{type(self).__name__} with either model_classes or model_objects"
            )

        if not model_params:
            model_params = {}
        self.set_params(model_params)

        if instantiate:
            self.__instantiate()

    def __get_self(self):
        if self.__instantiated:
            return self.model_objects
        else:
            return self.model_classes

    def __getitem__(self, key):
        return self.__get_self()[key]

    def __setitem__(self, key, value):
        self.__get_self()[key] = value

    def __iter__(self):
        return iter(self.__get_self())

    def __len__(self) -> int:
        return len(self.__get_self())

    def set_params(self, params):
        self.model_params.update(params)
        if self.__instantiated:
            for model_key, model_object in self.model_objects.items():
                model_params = params.get(model_key, {})
                model_object.set_params(model_params)

    def __instantiate(self, **additional_model_init_kwargs):
        if not self.__instantiated:
            self.model_objects = {}
            for target_name, model_class in self.model_classes.items():
                model_init_kwargs = {**self.model_params[target_name], **additional_model_init_kwargs}
                self.target_model_objects[target_name] = model_class(
                    **model_init_kwargs,
                )
        else:
            logger.info("Model objects already instantiated, updating params with additional_model_init_kwargs")
            for target_name, model_object in self.model_objects.items():
                model_params = additional_model_init_kwargs[target_name]
                model_object.set_params(model_params)
