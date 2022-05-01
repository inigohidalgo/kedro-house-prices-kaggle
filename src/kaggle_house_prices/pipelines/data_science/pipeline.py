"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""


from kedro.pipeline import Pipeline, node, pipeline
from kaggle_house_prices.pipelines.data_science import nodes as ds_nodes



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=ds_nodes.train_test_split,
                inputs=["model_input_table"],
                outputs=["df_train", "df_test"],
            ),
            node(
                func=ds_nodes.get_model_class,
                inputs=["params:model_options.model_name"],
                outputs="model_class",
            ),
            node(
                func=ds_nodes.train_model_on_df,
                inputs=["df_train", "model_class", "params:model_options"],
                outputs="trained_model_object",
            ),
        ]
    )
