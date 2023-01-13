from pathlib import Path

import pytest
import pandas as pd

from counterfactuals import (
    Model,
    PytorchModel,
    TensorflowModel,
    SklearnModel,
)


class TestInitialize:
    """Test the Model class"""

    def test_initialize_Model_empty(self):
        """Test the Model class"""
        model = Model.Model()
        assert model.backend == "pytorch"
        assert model.name is None

    def test_load(self, get_dataframe):
        """Test the _load method"""
        pass

    @pytest.mark.parametrize(
        "backend, class_name",
        [
            ("pytorch", "PytorchModel"),
            ("tensorflow", "TensorflowModel"),
            ("sklearn", "SklearnModel"),
        ],
    )
    def test_decide_implementation(self, backend, class_name):
        """Test getting the right backend implementation method
        Given a backend in string format,
        when the model is initialized,
        the correct backend implementation is returned confirmed through the class name
        """
        model = Model.Model(backend=backend)
        assert isinstance(model._implementation, eval(f"{class_name}.{class_name}"))


class TestTrain:

    pass


class TestEvaluate:

    pass
