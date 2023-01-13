from pathlib import Path
from typing import Optional, List, overload, Union, Any, Type

import pandas as pd
import sklearn

from counterfactuals.constants import Backend
from counterfactuals.BaseModel import BaseModel


class Model:
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
        self._implementation = self._get_implementation()

    def load(
        self,
        source: Union[Path, BaseModel, None] = None,
    ) -> None:
        """Load the model into the class
        Parameters
        ----------
        path (Path) : path to the model
        model () : the model to load
        """
        self._implementation.load(source)

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

            return SklearnModel()
        elif self.backend == Backend.tensorflow:
            from counterfactuals.TensorflowModel import TensorflowModel

            return TensorflowModel()
        elif self.backend == Backend.pytorch:
            from counterfactuals.PytorchModel import PytorchModel

            return PytorchModel()
        else:
            raise ValueError(f"Invalid Backend: {self.backend!r} not supported")
