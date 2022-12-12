from pathlib import Path
from typing import Optional, List, overload, Union, Any

import pandas as pd
import sklearn

from counterfactuals.constants import Backend

class Model:
    """Class for the model that is being explained."""

    def __init__(
        self,
        model: Optional[Any] = None,
        backend: Optional[str] = Backend.pytorch,
        name: Optional[str] = None,
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
        self.get_model_implementation(model, self.backend, self.name)

    def get_model_implementation(self, model, backend: Backend, name: str) -> Any:
        """Get the model implementation
        Parameters
        ----------
        backend (Backend) : the backend implementation
        Returns
        -------
        Any : the model implementation
        """
        self.__class__ = get_implementation(backend)
        self.__init__(model, backend=backend, name=name)

    def load(                           # type: ignore
        self,
        source: Union[Path, "Model", None] = None,
    ) -> None:
        """Load the model into the class
        Parameters
        ----------
        path (Path) : path to the model
        model () : the model to load
        """
        if isinstance(source, Path):
            raise NotImplementedError("Loading from path not implemented")  # TODO
        self._model = source


def get_implementation(backend: Backend) -> Model:
    """Get the implementation of the model
    Parameters
    ----------
    backend (str) : the backend implementation of the model
    Returns
    -------
    Model : the model implementation
    """
    if backend == Backend.sklearn:
        from counterfactuals.sklearnModel import SklearnModel

        return SklearnModel
    elif backend == Backend.tensorflow:
        from counterfactuals.TensorflowModel import TensorflowModel

        return TensorflowModel
    elif backend == Backend.pytorch:
        from counterfactuals.PytorchModel import PytorchModel

        return PytorchModel
    else:
        raise ValueError(f"Invalid Backend: {backend!r} not supported")
