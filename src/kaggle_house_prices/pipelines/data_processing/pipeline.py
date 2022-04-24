"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import raw_to_intermediate_houses


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=raw_to_intermediate_houses,
                inputs="house_prices",
                outputs="intermediate_house_prices",
            )
        ]
    )
