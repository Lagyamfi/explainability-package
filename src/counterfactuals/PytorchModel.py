from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from counterfactuals import Model
from counterfactuals.BaseModel import BaseModel
from counterfactuals.utils import conf_matrix


class PytorchModel(BaseModel):
    def __init__(
        self,
        model=None,
        backend: str = "pytorch",
        name: str = "",
    ) -> None:
        super().__init__(backend=backend, name=name)
        self._model: Optional[torch.nn.Module] = model
        self._train_x: pd.DataFrame = None
        self._train_y: pd.DataFrame = None
        self._test_x: pd.DataFrame = None
        self._test_y: pd.DataFrame = None
        self._predictions: pd.DataFrame = None
        self.name = name
        self._trained = False

    def load(
        self,
        source: Union[Path, BaseModel] = None,
        state_dict: bool = True,
    ) -> None:
        """Load the model into the class
        Parameters
        ----------
        path (Path) : path to the model
        model (torch.nn.Module) : the model to load
        """
        if isinstance(source, Path):
            source = torch.load(source)
            assert isinstance(source, torch.nn.Module), "Model must be a pytorch model"
        if not isinstance(source, torch.nn.Module):
            raise ValueError("Model must be a pytorch model")
        self._model = source

    def set_up(
        self,
        input_dim: Union[int, Tuple[int, int]],
        *args: Optional[List[int]],
        output_dim: int = 2,
        model_type: str = "MLP",
    ) -> None:
        """
        Set up the model
        Parameters
        ----------
        input_dim (int) : the input dimension
        args (List[int]) : the hidden layer dimensions
        output_dim (int) : the output dimension
        """
        # TODO: add more model types or define in constants file
        if self._model is not None:
            raise ValueError("Model already set up")
        assert model_type in [
            "MLP",
            "CNN",
        ], f"Invalid model type : {model_type!r}, must be MLP or CNN"
        if model_type == "MLP" and isinstance(input_dim, int):
            self._model = MLP(input_dim, *args, output_dim=output_dim, name=self.name)
        elif model_type == "CNN" and isinstance(input_dim, tuple):
            self._model = CNN(input_dim, output_dim=output_dim, name=self.name)
        else:
            raise ValueError(f"Invalid model type {model_type!r}")

    def trainer(
        self,
        train_data: pd.DataFrame = None,
        train_labels: pd.DataFrame = None,
        val_data: Optional[pd.DataFrame] = None,
        val_labels: Optional[pd.DataFrame] = None,
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
                   (see MLP implementation for details)

        """
        if self._model is None:
            raise ValueError("No model set up or loaded")
        if (train_data is None) and (train_labels is None):
            raise ValueError("No training data provided")
        if self._trained and not retrain:
            raise ValueError("Model already trained")
        try:
            self._model.train_model(
                train_data, train_labels, val_data, val_labels, **kwargs
            )
            self._trained = True
        except Exception as e:
            self._trained = False
            raise

    def predict(self, data: pd.DataFrame) -> torch.Tensor:
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
        if data is None:
            raise ValueError("No data provided")
        return self._model.predict(data)

    # @conf_matrix
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

        eval = self._model.evaluate(data, labels)
        return dict(zip(["accuracy", "predictions", "loss"], eval))

    def save(self, path: Path) -> None:
        """
        Save the model
        Parameters
        ----------
        path (Path) : the path to save the model to
        """
        if self._model is None:
            raise ValueError("No model set up or loaded")
        # torch.save(self._model.state_dict(), path) # TODO: save as recommended with state_dict
        torch.save(self._model, path)

    def __call__(self, test_data):
        return self.predict(test_data)


class MLP(torch.nn.Module):
    """
    A simple multi-layer perceptron
    """

    def __init__(
        self, input_dim: int, *hidden_dim: Any, output_dim: int, name: Optional[str]
    ) -> None:
        """
        Initialize the model
        Parameters
        ----------
        input_dim (int) : the input dimension
        *hidden_dim (Any) : the hidden dimensions
        output_dim (int) : the output dimension
        """
        super().__init__()
        self.name = name

        self.layers = torch.nn.ModuleList()
        if hidden_dim is not None:
            for _, dim in enumerate(hidden_dim, 1):
                self.layers.append(torch.nn.Linear(input_dim, dim))
                input_dim = dim
                self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(input_dim, output_dim))
        self.layers.append(torch.nn.Sigmoid())
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self._score = None

    @staticmethod
    def get_loader(
        data: Union[pd.DataFrame, np.ndarray], labels: pd.DataFrame, batch_size: int
    ) -> torch.utils.data.DataLoader:
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
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels.values, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(data, labels)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        return loader

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
        verbose: bool = True,
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
        train_loader = MLP.get_loader(train_x, train_y, batch_size)
        if not ((val_x is None) or (val_y is None)):
            val_loader = MLP.get_loader(val_x, val_y, batch_size)

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
                        print(
                            "Train Epoch: {}  Loss: {:.6f}".format(
                                epoch + 1, loss.data.item()
                            )
                        )

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
        self.eval()
        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()
        data = torch.tensor(data, dtype=torch.float32)
        y_pred = torch.max(self(data), dim=-1, keepdim=True)
        # return y_pred[1].detach().numpy()
        return y_pred[1]

    def evaluate(
        self, data: pd.DataFrame, labels: pd.DataFrame
    ) -> Tuple[float, np.ndarray, float]:
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
        self.eval()
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels.values, dtype=torch.long)
        y_pred = self(data)
        loss = float(self.loss_fn(y_pred, labels).detach())
        y_pred = torch.max(y_pred, dim=1, keepdim=True)
        accuracy = accuracy_score(labels, y_pred[1])
        return accuracy, y_pred[1].detach().numpy(), loss


class CNN(MLP):
    """
    A simple CNN model
    """

    def __init__(
        self,
        input_dim: Tuple[int, int] = (1, 32),
        hidden_dim: Optional[Tuple[int, ...]] = None,
        output_dim: int = 10,
        name: str = "",
    ) -> None:
        """
        Initialize the model
        Parameters
        ----------
        input_dim (Tuple[int, int, int]) : the input dimension
        *hidden_dim (Any) : the hidden dimensions
        output_dim (int) : the output dimension
        """
        super().__init__(input_dim[0], output_dim=output_dim, name=name)
        # TODO: make this more flexible and not hardcode the layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim[0], input_dim[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim[1], 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim),
        )

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
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x

    def train_model(
        self,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        val_x: Optional[pd.DataFrame] = None,
        val_y: Optional[pd.DataFrame] = None,
        epochs: int = 10,
        lr: float = 0.001,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        """
        Train the model
        Parameters
        ----------
        train_x (pd.DataFrame) : the training data
        train_y (pd.DataFrame) : the training labels
        val_x (pd.DataFrame) : the validation data
        val_y (pd.DataFrame) : the validation labels
        lr (float) : the learning rate
        epochs (int) : the number of epochs
        batch_size (int) : the batch size
        verbose (bool) : whether to print the loss
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # reshape the data to be (batch_size, channels, height, width)
        train_x = train_x.values.reshape(-1, 1, 28, 28)
        if not ((val_x is None) or (val_y is None)):
            val_x = val_x.values.reshape(-1, 1, 28, 28)
        super().train_model(
            train_x,
            train_y,
            val_x,
            val_y,
            epochs,
            lr,
            batch_size,
            verbose,
        )

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
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = data.reshape(-1, 1, 28, 28)
        return super().predict(data)

    def evaluate(
        self, data: Union[pd.DataFrame, np.ndarray], labels: pd.DataFrame
    ) -> Tuple[float, np.ndarray, float]:
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
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = data.reshape(-1, 1, 28, 28)
        return super().evaluate(data, labels)
