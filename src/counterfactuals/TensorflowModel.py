from typing import Optional, Any

import pandas as pd

from counterfactuals.BaseModel import BaseModel

class TensorflowModel(BaseModel):
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
        super().__init__(backend=backend, name=name)
        self._model: Optional[Any] = model    # type: ignore
        self._train_x: pd.DataFrame = None
        self._train_y: pd.DataFrame = None
        self._test_x: pd.DataFrame = None
        self._test_y: pd.DataFrame = None
        self._predictions: pd.DataFrame = None

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
        pass

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the output of the model
        Parameters
        ----------
        data (pd.DataFrame) : the data to predict
        Returns
        -------
        pd.DataFrame : the predictions
        """
        pass

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
        pass
