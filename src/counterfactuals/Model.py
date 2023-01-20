from pathlib import Path
from typing import Optional, List, overload, Union, Any, Type

import pandas as pd
import sklearn

from counterfactuals.constants import Backend
from counterfactuals.BaseModel import BaseModel


class Model(BaseModel):
    """Class for the model that is being explained."""

    def __init__(
        self,
        name: Optional[str] = None,
        backend: Optional[Backend] = Backend.pytorch,
    ) -> None:
        """
        Parameters
        ----------
        backend (str) : the backend implementation of the the model
        name (str) : the name of the model
        """
        if backend not in Backend.ALL:
            raise ValueError(f"Invalid Backend: {backend!r} not supported")
        self.backend = backend
        self.name = name
        self._model = self._get_implementation()
        self._is_trained = False

    def load(
        self,
        source: Union[str, BaseModel, None] = None,
    ) -> None:
        """Load the model into the class
        Parameters
        ----------
        path (Path) : path to the model
        model () : the model to load
        """
        if isinstance(source, str):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File {source!r} does not exist")
            source = path
        self._model.load(source)

    def _get_implementation(self) -> BaseModel:
        """Get the implementation of the model
        Parameters
        ----------
        backend (str) : the backend implementation of the model
        Returns
        -------
        Model : the model implementation
        """
        if self.backend == Backend.sklearn:
            from counterfactuals.SklearnModel import SklearnModel

            return SklearnModel(name=self.name)
        elif self.backend == Backend.tensorflow:
            from counterfactuals.TensorflowModel import TensorflowModel

            return TensorflowModel(name=self.name)
        elif self.backend == Backend.pytorch:
            from counterfactuals.PytorchModel import PytorchModel

            return PytorchModel(name=self.name)
        else:
            raise ValueError(f"Invalid Backend: {self.backend!r} not supported")

    def trainer(
        self,
        train_data: pd.DataFrame = None,
        train_labels: pd.DataFrame = None,
        val_data: Optional[pd.DataFrame] = None,
        val_labels: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> None:
        """
        Train the model
        Parameters
        ----------
        train_data (pd.DataFrame) : the training data
        train_labels (pd.DataFrame) : the training labels
        val_data (pd.DataFrame) : the testing data
        val_labels (pd.DataFrame) : the testing labels
        **kwargs : keyword arguments to pass to the model
                    (see MLP implementation for details)
        """
        try:
            self._model.trainer(
                train_data, train_labels, val_data, val_labels, **kwargs
            )
            self._is_trained = True
        except Exception as e:
            print("Training failed: ", e)
            self._is_trained = False
            raise e

    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Predict the labels for the data
        Parameters
        ----------
        data (pd.DataFrame) : the data to predict
        Returns
        -------
        pd.DataFrame : the predictions
        """
        return self._model.predict(data, **kwargs)

    def evaluate(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        """
        Evaluate the model on the data
        Parameters
        ----------
        data (pd.DataFrame) : the data to evaluate
        labels (pd.DataFrame) : the labels to evaluate
        """
        return self._model.evaluate(data, labels)

    def set_up(self, *args, **kwargs) -> None:
        """
        Set up the model
        Parameters
        ----------
        **kwargs : keyword arguments to pass to the model
                    (see MLP implementation for details)
        """
        self._model.set_up(*args, **kwargs)

    def save(self, path: Path) -> None:
        """
        Save the model
        Parameters
        ----------
        path (Path) : the path to save the model
        """
        self._model.save(path)
