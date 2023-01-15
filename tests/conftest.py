from pathlib import Path
from unittest import mock

import pytest
from counterfactuals import Data
from counterfactuals import Model
import pandas as pd


@pytest.fixture(name="get_dataframe", scope="module")
def fixture_get_dataframe() -> pd.DataFrame:
    """
    Fixtures for testing the Data class
    Returns
    -------
    path(Path) : path to the data
    data(pd.DataFrame) : the data
    """
    path = Path(__file__).parent.parent / "data" / "test_train.csv"
    return path, pd.read_csv(path)


@pytest.fixture()
def mock_data():
    error = ValueError("Either path or data must be specified")
    with mock.patch.object(Data, "Data", side_effect=error, autospec=True) as mock_data:
        yield mock_data


@pytest.fixture(name="get_model", scope="module")
def fixture_model(backend: str = "pytorch"):
    """
    Fixtures for testing the Model class
    Returns
    -------
    model(Model) : the model
    """
    return Model.Model(name="test", backend=backend)
