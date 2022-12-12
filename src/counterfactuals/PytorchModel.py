from pathlib import Path
from typing import Optional, Union, Any, List, Tuple, Dict

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from counterfactuals.BaseModel import BaseModel
from counterfactuals.utils import conf_matrix

class PytorchModel(BaseModel):
    def __init__(self, model: Optional[torch.nn.Module] = None, backend: str = "pytorch", name: str = None) -> None:
        super().__init__(backend=backend, name=name)
        self._model: torch.nn.Module = model
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
        model (torch.nn.Module) : the model to load
        """
        if isinstance(source, Path):
            raise NotImplementedError("Loading from path not implemented")  # TODO
        if not isinstance(source, torch.nn.Module):
            raise ValueError("Model must be a pytorch model")
        self._model = source

    def set_up(self, input_dim: int, *args: Optional[List[int]], output_dim: int = 10) -> None:
        """
        Set up the model
        Parameters
        ----------
        input_dim (int) : the input dimension
        args (List[int]) : the hidden layer dimensions
        output_dim (int) : the output dimension
        """
        self._model = MLP(input_dim, *args, output_dim=output_dim)

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
        if ((train_data is None) and (train_labels is None)):
            raise ValueError("No training data provided")
        if self._model is None:
            raise ValueError("No model set up")
        self._model.train_model(train_data, train_labels, val_data, val_labels, **kwargs)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict the labels for the data
        Parameters
        ----------
        data (pd.DataFrame) : the data to predict
        Returns
        -------
        np.ndarray : the predictions
        """
        if self._model is None:
            raise ValueError("No model set up or loaded")

        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = torch.tensor(data.values, dtype=torch.float32)
        y_pred = torch.max(self._model(data), dim=1)
        return torch.tensor(y_pred[1].detach().numpy())

    @conf_matrix
    def evaluate(self, data: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the model on the data
        Parameters
        ----------
        data (pd.DataFrame) : the data to evaluate on
        labels (pd.DataFrame) : the labels to evaluate on
        Returns
        -------
        Tuple[float, np.ndarray, float] : the accuracy, predictions, and loss
        """
        if self._model is None:
            raise ValueError("No model set up or loaded")
        if data is None:
            raise ValueError("No data provided")
        if labels is None:
            raise ValueError("No labels provided")

        data = torch.tensor(data.values, dtype=torch.float32)
        labels = torch.tensor(labels.values, dtype=torch.long)
        y_pred = self._model(data)
        loss = float(self._model.loss_fn(y_pred, labels).detach())
        y_pred = torch.max(y_pred, dim=1)
        accuracy = accuracy_score(labels, y_pred[1])
        return dict(
            acc=accuracy,
            predictions=y_pred[1],
            loss=loss
        )

    def __call__(self, test_data):
        return self.predict(test_data)


def get_loader(data: pd.DataFrame, labels: pd.DataFrame, batch_size: int) -> torch.utils.data.DataLoader:
    """
    Get a data loader for the data
    Parameters
    ----------
    data (pd.DataFrame) : the data
    labels (pd.DataFrame) : the labels
    batch_size (int) : the batch size
    Returns
    -------
    torch.utils.data.DataLoader : the data loader
    """
    data = torch.tensor(data.values, dtype=torch.float32)
    labels = torch.tensor(labels.values, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(data, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


class MLP(torch.nn.Module):
    """
    A simple multi-layer perceptron
    """
    def __init__(self, input_dim: int, *hidden_dim: Any, output_dim: int) -> None:
        """
        Initialize the model
        Parameters
        ----------
        input_dim (int) : the input dimension
        *hidden_dim (Any) : the hidden dimensions
        output_dim (int) : the output dimension
        """
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()
        if hidden_dim is not None:
            for _, dim in enumerate(hidden_dim, 1):
                self.layers.append(torch.nn.Linear(input_dim, dim))
                input_dim = dim
                self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(input_dim, output_dim))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self._score = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Parameters
        ----------
        x (torch.Tensor) : the input
        Returns
        -------
        torch.Tensor : the output
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def train_model(
        self,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        val_x: Optional[pd.DataFrame] = None,
        val_y: Optional[pd.DataFrame] = None,
        epochs: int = 20,
        lr: float = 0.01,
        batch_size: int = 128,
        verbose: bool = True
    ) -> None:
        """
        Train the model
        Parameters
        ----------
        train_x (pd.DataFrame) : the training data
        train_y (pd.DataFrame) : the training labels
        val_x (pd.DataFrame) : the validation data
        val_y (pd.DataFrame) : the validation labels
        epochs (int) : the number of epochs
        lr (float) : the learning rate
        batch_size (int) : the batch size
        verbose (bool) : whether to print the loss
        """

        # define loss function
        loss_fn = self.loss_fn
        # define optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # create data loaders
        train_loader = get_loader(train_x, train_y, batch_size)
        if not ((val_x is None) or (val_y is None)):
            val_loader = get_loader(val_x, val_y, batch_size)

        # train the model
        for epoch in range(epochs):
            self.train()
            for batch_idx, (inputs, target) in enumerate(train_loader, 0):

                # Forward pass
                optimizer.zero_grad()
                y_pred = self(inputs)
                loss = loss_fn(y_pred, target)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # print statistics
                if verbose:
                    if batch_idx % 100 == 0:
                        print('Train Epoch: {}  Loss: {:.6f}'.format(
                              epoch + 1,
                              loss.data.item()))

        print("Training complete")

        if not ((val_x is None) or (val_y is None)):
            self.eval()
            accuracy, _, loss = self.evaluate(val_x, val_y)
            print("Validation accuracy: ", accuracy)
            print("Validation loss: ", loss)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict on the data
        Parameters
        ----------
        data (pd.DataFrame) : the data to predict on
        Returns
        -------
        np.ndarray : the predictions
        """
        data = torch.tensor(data.values, dtype=torch.float32)
        y_pred = torch.max(self(data), dim=1)
        return y_pred[1].detach().numpy()

    def evaluate(self, data: pd.DataFrame, labels: pd.DataFrame) -> Tuple[float, np.ndarray, float]:
        """
        Evaluate the model
        Parameters
        ----------
        data (pd.DataFrame) : the data
        labels (pd.DataFrame) : the labels
        Returns
        -------
        Tuple[float, np.ndarray, float] : the accuracy, predictions, and loss
        """
        data = torch.tensor(data.values, dtype=torch.float32)
        labels = torch.tensor(labels.values, dtype=torch.long)
        y_pred = self(data)
        loss = float(self.loss_fn(y_pred, labels).detach())
        y_pred = torch.max(y_pred, dim=1)
        accuracy = accuracy_score(labels, y_pred[1])
        return accuracy, y_pred[1].detach().numpy(), loss
