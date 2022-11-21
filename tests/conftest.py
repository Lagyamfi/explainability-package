from pathlib import Path

import pytest
from counterfactuals import Data
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
