"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""

from ast import FunctionDef
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_model_object, node_preprocess_df_for_model, node_train_model_on_Xy, train_test_split, get_model_object
from sklearn import model_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_test_split,
            inputs=["model_input_table", "params:model_options:train_test_split"],
            outputs=["df_train", "df_test"],
        ),
        node(
            func=node_preprocess_df_for_model,
            inputs=["df_train", "params:model_options:preprocess_params", "params:model_options:target_name"],
            outputs=["X_train", "y_train"],
        ),
        node(func=get_model_object,
                inputs=["params:model_options:model_name"],
                outputs=["model_object"],),
        )
        node(func=node_train_model_on_Xy,
        inputs=["X_train", "y_train", "model_object", "params:model_options"],
        outputs=["model_object"],),
        )


    ])
