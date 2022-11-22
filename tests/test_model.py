from pathlib import Path

import pytest
import pandas as pd

from counterfactuals import Model, Data

class TestInitialize:
    """Test the Model class"""

    def test_initialize_Model_empty(self):
        """Test the Model class"""
        model = Model.Model()
        assert model.backend == "sklearn"
        assert model.name is None

    def test_load(self, get_dataframe):
        """Test the _load method"""
        pass


class TestTrain:

    pass


class TestEvaluate:

    pass


