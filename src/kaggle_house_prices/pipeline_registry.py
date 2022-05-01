"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from kaggle_house_prices.pipelines import data_processing as dp, data_science as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_processing = dp.create_pipeline()
    data_science = ds.create_pipeline()

    return {
        "__default__": pipeline([]),
        "dp": data_processing,
        "ds": data_science,
    }
