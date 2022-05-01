"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from kaggle_house_prices.pipelines.data_processing import nodes as dp_nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=dp_nodes.raw_to_intermediate_houses,
                inputs="house_prices",
                outputs="intermediate_house_prices",
            ),
            node(
                func=dp_nodes.preprocess_df_for_model,
                inputs=["intermediate_house_prices", "params:model_options"],
                outputs="model_input_table",
            ),
        ]
    )
