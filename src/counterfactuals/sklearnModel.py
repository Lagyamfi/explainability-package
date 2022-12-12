from pathlib import Path
import abc
from typing import Optional, List, overload, Union
import enum

import pandas as pd
import sklearn

from counterfactuals.BaseModel import BaseModel

class SklearnModel(BaseModel):
    def __init__(
        self,
        model: Optional[sklearn.base.BaseEstimator] = None,
        backend: str = "sklearn", name: str = None
    ) -> None:
        super().__init__(backend=backend, name=name)
        self._model: sklearn.base.BaseEstimator = model
        self._train_x: pd.DataFrame = None
        self._train_y: pd.DataFrame = None
        self._test_x: pd.DataFrame = None
        self._test_y: pd.DataFrame = None
        self._predictions: pd.DataFrame = None

    def load(
        self,
        source: Union[Path, BaseModel] = None,
    ) -> None:
        """Load the model into the class
        Parameters
        ----------
        path (Path) : path to the model
        model (sklearn.base.BaseEstimator) : the model to load
        """
        if isinstance(source, Path):
            raise NotImplementedError("Loading from path not implemented")  # TODO
        if not isinstance(source, sklearn.base.BaseEstimator):
            raise ValueError("Model must be a sklearn model")
        self._model = source

    def train(
        self,
        train_data: pd.DataFrame = None,
        test_data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Train the model
        Parameters
        ----------
        train_data (pd.DataFrame) : the training data
        test_data (pd.DataFrame) : the testing data
        """
        if not (self.train_x and self.train_y):
            raise ValueError("No training data provided")
        raise NotImplementedError("Training not implemented")  # TODO
