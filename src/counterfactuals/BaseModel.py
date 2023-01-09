from pathlib import Path
import abc
from typing import Optional, List, overload, Union, Any
import enum

import pandas as pd
import sklearn

from counterfactuals.constants import Backend

class BaseModel:
    """Class for the model that is being explained."""

    def __init__(
        self,
        model: Optional[Any] = None,
        backend: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        backend (str) : the backend implementation of the the model
        name (str) : the name of the model
        """
        try:
            self.backend = backend
        except ValueError:
            raise ValueError(f"Invalid Backend: {backend!r} not supported")
        self.name = name
        self._model: Optional[Any] = model    # type: ignore
        self._train_x: pd.DataFrame = None
        self._train_y: pd.DataFrame = None
        self._test_x: pd.DataFrame = None
        self._test_y: pd.DataFrame = None
        self._predictions: pd.DataFrame = None

    @overload
    def load(
        self,
        path: Optional[Path] = None,
    ) -> None:
        """Load the model into the class
        Parameters
        ----------
        path (Path) : path to the model
        """
        ...

    @overload
    def load(
        self,
        model: Optional["BaseModel"] = None,
    ) -> None:
        """Load the model into the class
        Parameters
        ----------
        model ("Model") : the model to load
        """
        ...

    def load(                           # type: ignore
        self,
        source: Union[Path, "BaseModel", None] = None,
    ) -> None:
        """Load the model into the class
        Parameters
        ----------
        path (Path) : path to the model
        model (sklearn.base.BaseEstimator) : the model to load
        """
        if isinstance(source, Path):
            raise NotImplementedError("Loading from path not implemented")  # TODO
        self._model = source

    @abc.abstractmethod
    def trainer(
        self,
    ) -> None:
        """
        Train the model
        Parameters
        ----------
        train_data (pd.DataFrame) : the training data
        test_data (pd.DataFrame) : the testing data
        """
        ...

    @abc.abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the output of the model
        Parameters
        ----------
        data (pd.DataFrame) : the data to predict
        Returns
        -------
        pd.DataFrame : the predictions
        """
        ...

    @abc.abstractmethod
    def evaluate(
        self,
        data_x: pd.DataFrame,
        data_y: pd.DataFrame,
        return_df: Optional[bool] = None,
        conf_mat: Optional[bool] = None
    ) -> pd.DataFrame:
        """Evaluate the model
        Parameters
        ----------
        data (pd.DataFrame) : the data to evaluate
        return_df (bool) : whether to return a dataframe of the results
        conf_mat (bool) : whether to return a confusion matrix

        Returns
        -------
        pd.DataFrame : the evaluation results
        """
        ...

    def __repr__(self) -> str:
        return (
            f"Model(backend={self.backend}, name={self.name}, "
            f"model={self._model.__class__.__name__})"
        )
