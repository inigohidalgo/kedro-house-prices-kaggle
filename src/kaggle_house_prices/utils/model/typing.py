from typing import Union, Number, TypeVar, Mapping, Any, Protocol, runtime_checkable

from kaggle_house_prices.utils.data.typing import KeyT

ModelCfg = Mapping[str, Any]
ModelCfgMapping = Mapping[KeyT, ModelCfg]


@runtime_checkable
class AbstractModel(Protocol):
    def predict(self, data, **predict_kwargs):
        ...

    def fit(self, data, **fit_kwargs):
        ...

    def set_params(self, **params):
        ...

ModelT = TypeVar("ModelT", bound=AbstractModel)
ModelCls = TypeVar("ModelCls")