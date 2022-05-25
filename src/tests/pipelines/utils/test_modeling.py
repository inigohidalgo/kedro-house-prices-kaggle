from __future__ import annotations
from kaggle_house_prices.utils import modeling as modeling_utils

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

def test_inheritance():
    class TestFitter:
        def fit(self, X, y, **kwargs):
            ...

    class TestPredictor:
        def predict(self, X, **kwargs):
            ...

    class TestModel(TestFitter, TestPredictor):
        ...

    assert not issubclass(TestFitter, modeling_utils.AbstractGenericModel)
    assert not issubclass(TestPredictor, modeling_utils.AbstractGenericModel)
    assert issubclass(TestModel, modeling_utils.AbstractGenericModel)

def test_target_model():
    class TestFitter:
        def fit(self, X, y, **kwargs):
            ...

    class TestPredictor:
        def predict(self, X, **kwargs):
            ...

    class TestModel(TestFitter, TestPredictor):
        ...
    
    test_target_model = modeling_utils.TargetModel(TestModel, "target")
    test_data = pd.DataFrame({"target": [1, 2, 3], "col1": [1, 2, 3]})
    test_X, test_y = test_target_model.get_Xy_from_data(test_data)
    expected_X = pd.DataFrame({"col1": [1, 2, 3]})
    expected_y = pd.Series([1, 2, 3], name="target")
    assert_frame_equal(test_X, expected_X)
    assert_series_equal(test_y, expected_y)