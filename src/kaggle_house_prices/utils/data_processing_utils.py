import pandas as pd
import numpy as np
import logging

log = logging.getLogger()

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