"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""

import pandas as pd
import numpy as np
import logging
from kaggle_house_prices.utils import date_utils as dt_utils, data_processing_utils as dp_utils

log = logging.getLogger()


def raw_to_intermediate_houses(houses_dataset):
    houses_dataset["CentralAir"] = dp_utils._is_true(houses_dataset["CentralAir"])
    houses_dataset["DtSold"] = dt_utils.get_date_from_columns(houses_dataset, {"year": "YrSold", "month": "MoSold"})
    return houses_dataset
