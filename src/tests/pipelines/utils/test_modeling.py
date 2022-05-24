from __future__ import annotations
from kaggle_house_prices.utils import modeling as modeling_utils


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
