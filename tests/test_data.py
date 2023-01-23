from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from counterfactuals.Data import Data


class TestInitialize:
    """Test the Data class"""

    def test_initialize_Data_empty(self, mock_data):
        """Test the Data class"""
        with pytest.raises(ValueError or TypeError) as info:  # type: ignore
            data = mock_data()
        expected = "Either path or data must be specified"  # TODO probably should be a constant
        assert expected in str(info.value)

    def test_initialize_with_path(self, get_dataframe):
        """Test initializing the Data class with a path"""
        path = get_dataframe[0]
        data = Data(path)
        assert data.name is None
        assert data.dataframe.shape == (99, 785)
        # check access to train_data raises error
        with pytest.raises(ValueError) as info:
            data.train_data
        expected = "No training data, perform split on Data!"
        assert expected in str(info.value)

    def test_initialize_with_dataframe(self, get_dataframe):
        """Test initializing the Data class with dataframe"""
        data = Data(get_dataframe[1], name="test")
        assert data.name == "test"
        assert data.dataframe.shape == (99, 785)
        with pytest.raises(ValueError) as info:
            data.train_data
        expected = "No training data, perform split on Data!"
        assert expected in str(info.value)

    def test_load(self, get_dataframe):
        """Test the _load method"""
        data = Data(source=get_dataframe[0])
        data._load(get_dataframe[1])
        assert data.dataframe.shape == (99, 785)


class TestPrint:
    """Test the __repr__ method"""

    def test_print(self, get_dataframe):
        """Test the __repr__ method"""
        data = Data(get_dataframe[1], name="test")
        expected = "Data(name='test', dataframe shape=(99, 785))"
        assert data.__repr__() == expected


class TestSplit:
    def test_data_split_default(self, get_dataframe):
        """Test the split method with default parameters"""
        data = Data(get_dataframe[1], name="test")
        data.split()
        assert data.train_data[0].shape == (79, 784)
        assert data.train_data[1].shape == (79,)
        assert data.val_data[0].shape == (20, 784)
        assert data.val_data[1].shape == (20,)

    def test_data_split_custom(self, get_dataframe):
        """Test the split method with custom parameters"""
        data = Data(get_dataframe[1], name="test")
        data.split(split_size=0.1)
        assert data.train_data[0].shape == (89, 784)
        assert data.train_data[1].shape == (89,)
        assert data.val_data[0].shape == (10, 784)
        assert data.val_data[1].shape == (10,)

    def test_data_split_no_data(self, get_dataframe):
        """Test the split method with no data"""
        data = Data(get_dataframe[0])
        # data._dataframe = None
        with mock.patch.object(
            data, "_dataframe", None
        ):  # done be able to test split method error
            with pytest.raises(ValueError) as info:
                data.split()
            expected = "No data to split"
            assert expected in str(info.value)


class TestPCA:
    def test_pca_default(self, get_dataframe):
        """Test the pca method"""
        data = Data(get_dataframe[1], name="test")
        data.split()
        data.pca()
        assert data._pca_train_x.shape == (79, 2)
        assert data._pca_val_x.shape == (20, 2)

    def test_pca_custom(self, get_dataframe):
        """Test the pca method with custom parameters"""
        data = Data(get_dataframe[1], name="test")
        data.split()
        data.pca(n_components=20)
        assert data._pca_train_x.shape == (79, 20)
        assert data._pca_val_x.shape == (20, 20)

    def test_pca_already_done(self, get_dataframe):
        """Test the pca method with custom parameters"""
        data = Data(get_dataframe[1], name="test")
        data.split()
        data.pca(n_components=10)
        with pytest.raises(ValueError) as info:
            data.pca(n_components=10)
        expected = "PCA already performed"
        assert expected in str(info.value)

    @pytest.mark.parametrize("split_type", ["dataframe", "training", "validation"])
    def test_pca_data(self, get_dataframe, split_type):
        """Test the pca method with no data"""
        data = Data(get_dataframe[1], name="test")
        setattr(data, split_type, None)  # done be able to test pca method error
        with pytest.raises(ValueError) as info:
            data.pca()
        expected = "No data to perform PCA"
        assert expected in str(info.value)
