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


@pytest.fixture(name="get_data", scope="class")
def fixture_get_data(get_dataframe):
    """
    Fixtures for testing the Data class
    Returns
    -------
    data(Data) : the data
    """
    data = Data.Data(get_dataframe[1], name="test_data")
    data.split(0.2)
    data.pca(10)
    return data


@pytest.fixture()
def mock_data():
    error = ValueError("Either path or data must be specified")
    with mock.patch.object(Data, "Data", side_effect=error, autospec=True) as mock_data:
        yield mock_data


@pytest.fixture(name="get_model", scope="function")
def fixture_model(backend: str = "pytorch"):
    """
    Fixtures for testing the Model class
    Returns
    -------
    model(Model) : the model
    """
    return Model.Model(name="test", backend=backend)


@pytest.fixture(name="set_model", scope="function")
def fixture_set_model(get_model):
    """
    Fixtures for testing the Model class
    Returns
    -------
    model(Model) : the model
    """
    trained = get_model
    input_dim = 10
    output_dim = 10
    trained.set_up(input_dim=input_dim, output_dim=output_dim)
    return trained


@pytest.fixture(name="get_trained_model", scope="function")
def fixture_trained_model(set_model, get_data):
    """
    Fixtures for testing the Model class
    Returns
    -------
    model(Model) : the model
    """
    data = get_data
    trained = set_model
    trained.trainer(*data.train_data, epochs=1)
    return trained
