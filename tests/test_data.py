from pathlib import Path

import pytest
import pandas as pd

from counterfactuals.Data import Data

class TestInitialize:
    """Test the Data class"""

    def test_initialize_Data_empty(self):
        """Test the Data class"""
        with pytest.raises(ValueError) as info:
            data = Data()
        expected = "Either path or data must be specified"  # TODO probably should be a constant
        assert expected in str(info.value)

    def test_initialize_with_path(self, get_dataframe):
        """Test initializing the Data class with a path"""
        path = get_dataframe[0]
        data = Data(path=path)
        assert data.path == path
        assert data.name == None
        assert data.dataframe.shape == (99, 785)
        # assert data.training.shape == (80, 3)
        # assert data.testing.shape == (20, 3)

    def test_initialize_with_dataframe(self, get_dataframe):
        """Test initializing the Data class with dataframe"""
        data = Data(data=get_dataframe[1])
        assert data.path == None
        assert data.name == None
        assert data.dataframe.shape == (99, 785)


