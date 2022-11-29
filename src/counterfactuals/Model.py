from pathlib import Path
import abc
from typing import Optional, List, overload, Union
import enum

import pandas as pd
import sklearn

class Backend(enum.Enum):
    """The backend to use for the model"""
    sklearn = "sklearn"
    tensorflow = "tensorflow"
    pytorch = "pytorch"


class Model(abc.ABC):
    """Class for the model that is being explained."""

    def __init__(
        self,
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
            self.backend = Backend(backend)
        except ValueError:
            raise ValueError(f"Invalid Backend: {backend!r} not supported")
        self.name = name
        self._model: Optional["Model"] = None       # type: ignore
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
        model: Optional["Model"] = None,
    ) -> None:
        """Load the model into the class
        Parameters
        ----------
        model ("Model") : the model to load
        """
        ...

    def load(                           # type: ignore
        self,
        source: Union[Path, "Model", None] = None,
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
    def train(
        self,
        train_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Train the model
        Parameters
        ----------
        train_data (pd.DataFrame) : the training data
        test_data (pd.DataFrame) : the testing data
        """
        ...

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the output of the model
        Parameters
        ----------
        data (pd.DataFrame) : the data to predict
        Returns
        -------
        pd.DataFrame : the predictions
        """
        return self._model.predict(data)

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
        self._predictions = self.predict(data_x)

        print(f"{self._model.score(data_x, data_y) * 100 :.2f} % ")
        if conf_mat:
            disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(data_y, self._predictions)
            disp.figure_.suptitle("Confusion Matrix")
        if return_df:
            return self._predictions

    def get_queries(
        self,
        true_labels,
        predicted: Optional[float] = None,
        expected: Optional[float] = None
    ) -> pd.DataFrame:
        """Get queries where predicted != expected
        Parameters
        ----------
        predicted (float) : the predicted value
        expected (float) : the expected value
        Returns
        -------
        pd.DataFrame : the queries
        """
        test_df = pd.DataFrame([self._predictions, true_labels]).T
        test_df.columns = ['predictions', 'true_labels']
        condition = f"(predictions != true_labels) & (true_labels == {expected}) & (predictions == {predicted})"
        query_list = test_df.query(condition).index
        return query_list

    def __repr__(self) -> str:
        return (
            f"Model(backend={self.backend}, name={self.name}, "
            f"model={self._model.__class__.__name__})"
        )
