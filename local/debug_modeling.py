from kaggle_house_prices.utils import modeling as modeling_utils
from typing import Optional, Protocol, Union, Dict, Any, TypeVar, Generic
from sklearn import dummy
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


dumb = modeling_utils.Model(model_class=dummy.DummyRegressor, instantiate=True)

data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

DataT = Union[dict, tuple]
DataHandlingModeT = TypeVar("DataHandlingModeT", Literal["train"], Literal["predict"])


# TODO: is this too restrictive a protocol/signature for datahandling
class DataHandler(Protocol):
    """
    Takes a single data object and formats it for training or prediction.
    """

    def __call__(self, data: DataT, mode: DataHandlingModeT) -> DataT:
        ...


class DataHandler:
    def __init__(self, **kwargs: Dict[str, Any])->None:
        if "data" in kwargs:
            self.receive_data(kwargs["data"])
        
    def receive_data(self, data: pd.DataFrame)->None:
        self.data = data
    
    def emit_data(self):
        return self.data
    
    def transform_data(self)->None:
        
        pass

    def __call__(self, data: Any):



