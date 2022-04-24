"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from kaggle_house_prices.pipelines import data_processing as dp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    raw_to_intermediate_pipeline = dp.create_pipeline()

    return {
        "__default__": pipeline([]),
        "raw_to_intermediate": raw_to_intermediate_pipeline,
    }
