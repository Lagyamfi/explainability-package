from pathlib import Path
import abc
from typing import Optional, List, overload, Union
import enum

import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from counterfactuals.BaseModel import BaseModel

MODELS = {
    "logistic": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "svm": NuSVC,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "mlp": MLPClassifier,
}


class SklearnModel(BaseModel):
    def __init__(
        self,
        model: Optional[sklearn.base.BaseEstimator] = None,
        backend: str = "sklearn",
        name: str = "",
    ) -> None:
        super().__init__(backend=backend, name=name)
        self._model: sklearn.base.BaseEstimator = None
        self._train_x: pd.DataFrame = None
        self._train_y: pd.DataFrame = None
        self._test_x: pd.DataFrame = None
        self._test_y: pd.DataFrame = None
        self._predictions: pd.DataFrame = None
        self._trained = False

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

    def set_up(
        self,
        model_type: str = "logistic",
    ) -> None:
        """Set up the model
        Parameters
        ----------
        model_type (str) : the type of model to use
        """
        if self._model is not None:
            raise ValueError("Model already set up")
        assert model_type in MODELS, f"Invalid model type: {model_type!r}"
        self._model = MODELS[model_type]()

    def trainer(
        self,
        train_data: pd.DataFrame = None,
        train_labels: pd.DataFrame = None,
        val_data: pd.DataFrame = None,
        val_labels: pd.DataFrame = None,
        retrain: bool = False,
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

        """
        if train_data is None or train_labels is None:
            raise ValueError("No training data provided")
        if not retrain and self._trained:
            raise ValueError("Model already trained")
        try:
            self._model.fit(train_data, train_labels, **kwargs)
            self._trained = True
        except Exception as e:
            self._trained = False
            raise

    def predict(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """Predict the labels for the data
        Parameters
        ----------
        data (pd.DataFrame) : the data to predict
        **kwargs : keyword arguments to pass to the model
        Returns
        -------
        pd.DataFrame : the predictions
        """
        if not self._trained:
            raise ValueError("Model not trained")
        if data is None:
            raise ValueError("No data provided")
        if kwargs.get("return_proba", False):
            return self._model.predict_proba(data)
        return self._model.predict(
            data,
        )

    def evaluate(self, data: pd.DataFrame = None, labels: pd.DataFrame = None) -> float:
        """Evaluate the model
        Parameters
        ----------
        data (pd.DataFrame) : the data to evaluate
        labels (pd.DataFrame) : the labels to evaluate
        Returns
        -------
        float : the score of the model
        """
        if not self._trained:
            raise ValueError("Model not trained")
        if data is None:
            raise ValueError("No data provided")
        if labels is None:
            raise ValueError("No labels provided")
        return self._model.score(data, labels)
