from typing import Tuple, Union

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


def _is_true(series: pd.Series, true_key="Y", false_key=None) -> pd.Series:
    log.info("Wasupp")
    conditions = [series == true_key]
    choices = [True, False]
    if false_key is not None:
        conditions.append(series == false_key)
    else:
        conditions.append(True)
    true_values = np.select(conditions, choices, default=np.nan)
    true_series = pd.Series(true_values, index=series.index, name=series.name).astype(bool)

    return true_series


def extract_target_from_df(df, target_name=None, return_Xy=True) -> Union[Tuple[pd.DataFrame, pd.Series], pd.Series]:
    if target_name is None:
        target_name = "target"
        log.info(f"No target name specified, assuming default: {target_name}")
    log.info(f"Extracting target {target_name} from df")
    target_series = df[target_name]
    if return_Xy:
        log.debug("Returning df and target as (X, y)")
        return df.drop(columns=target_name), target_series
    else:
        log.debug("Returning target variable as Series")
        return df[target_name]
