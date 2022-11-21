from pathlib import Path
from typing import Optional, List

import pandas as pd
from sklearn.model_selection import train_test_split


class Data:
    """A class for the data used in the experimets"""

    def __init__(
        self,
        path: Path = None,
        data: pd.DataFrame = None,
        name: str = None) -> None:
        """
        Parameters
        ----------
        path (Path) : path to the data
        name (str)  : name of the dataset
        """
        self.path = path
        self.name = name
        self._training: pd.DataFrame = None
        self._testing: pd.DataFrame = None
        self._dataframe: pd.DataFrame = None
        if isinstance(data, pd.DataFrame):
            self.load(data)
        elif isinstance(path, Path):
            self.load(pd.read_csv(path))
        else:
            raise ValueError("Either path or data must be specified")
        self.feature_names: List[str] = None
        self.target_name: str = None
        self._pca_train: Optional[pd.DataFrame] = None
        self._pca_test: Optional[pd.DataFrame] = None

    def split(self, split_size: float = 0.2) -> None:
        """Split the data into training and testing sets
        Parameters
        ----------
        split_size (float) : the size of the testing set

        Returns
        -------
        None
        """
        self._training, self._testing = train_test_split(
            self._dataframe, test_size=split_size
        )

    def load(self, data: pd.DataFrame, split: bool = True) -> None:
        """Load the data into the class and split it if necessary
        Parameters
        ----------
        data (pd.DataFrame) : the data to load
        split (bool)        : whether to split the data into training and testing sets

        Returns
        -------
        None
        """
        self.dataframe = data     # TODO validate data
        if split:
            self.split()

    def pca(self, n_components: int = 2) -> None:
        """Perform PCA on the data
        Parameters
        ----------
        n_components (int) : the number of components to use

        Returns
        -------
        None
        """
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        self._pca_train = pca.fit_transform(self.training)
        self._pca_test = pca.transform(self.testing)

    @property
    def training(self, pca: bool = None) -> pd.DataFrame:
        """The training set
        Parameters
        ----------
        pca (bool) : whether to return the PCA transformed data

        Returns
        -------
        pd.DataFrame
            the training set

        Raises
        ------
        ValueError
            if the training set has not been loaded
        """
        # TODO exception handling data loaded
        if pca:
            return self._pca_train
        return self._training

    @training.setter
    def training(self, data: pd.DataFrame) -> None:
        self._training = data

    @property
    def testing(self, pca: bool = None) -> pd.DataFrame:
        """The testing set
        Parameters
        ----------
        pca (bool) : whether to return the PCA transformed data

        Returns
        -------
        pd.DataFrame
            the testing set

        Raises
        ------
        ValueError
            if the testing set has not been loaded
        """
        # TODO exception handling data loaded
        if pca:
            return self._pca_test
        return self._testing

    @testing.setter
    def testing(self, data: pd.DataFrame) -> None:
        self._testing = data

    @property
    def dataframe(self) -> pd.DataFrame:
        """The entire dataset
        Returns
        -------
        pd.DataFrame
            the entire dataset

        Raises
        ------
        ValueError
            if the dataset has not been loaded
        """
        # TODO exception handling data loaded
        return self._dataframe

    @dataframe.setter
    def dataframe(self, data: pd.DataFrame) -> None:
        self._dataframe = data

    def __repr__(self) -> str:
        return f"Data({self.name}, {self.path})"
