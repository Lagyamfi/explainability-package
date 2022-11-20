from pathlib import Path

from counterfactuals.Data import Data


def test_Data():
    """Test the Data class"""
    data = Data()
    assert data.path is None
    assert data.name is None


def test_load_from_path():
    """Test loading data from a path"""
    path = Path("../data/train.csv")
    data = Data(path=path)
    assert data.path == path
    assert data.name == None
    #assert data.dataframe.shape == (100, 3)
    #assert data.training.shape == (80, 3)
    #assert data.testing.shape == (20, 3)