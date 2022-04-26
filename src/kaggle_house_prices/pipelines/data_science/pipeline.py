"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_df_for_model, train_model_on_df, train_test_split
from sklearn import model_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_test_split,
            inputs=["model_input_table", "params:model_options:train_test_split"],
            outputs=["df_train", "df_test"],
        )

    ])
