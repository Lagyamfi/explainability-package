from pathlib import Path
from typing import Optional, List, Union, overload, Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class Data:
    """A class for the data used in the experimets"""

    @overload
    def __init__(
        self, path: Path, *, name: Optional[str] = None
    ) -> None:  # pragma: no cover
        ...

    @overload
    def __init__(
        self, data: pd.DataFrame, *, name: Optional[str] = None
    ) -> None:  # pragma: no cover
        ...

    def __init__(  # type: ignore
        self,
        source: Union[Path, pd.DataFrame, None] = None,
        *,
        target_name: str = "label",
        name: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        source (Union[Path, pd.DataFrame]) : the source of the data
        name (str)  : name of the dataset
        """
        self.name = name
        self._train_x: pd.DataFrame = None
        self._train_y: pd.DataFrame = None
        self._val_x: Optional[pd.DataFrame] = None
        self._val_y: Optional[pd.DataFrame] = None
        self._dataframe: pd.DataFrame = None
        self.feature_names: Optional[List[str]] = None
        self.target_name: str = target_name
        self._pca_train_x: Optional[pd.DataFrame] = None
        self._pca_object: Optional[Any] = None
        self._pca_val_x: Optional[pd.DataFrame] = None
        if source is not None:
            if isinstance(source, pd.DataFrame):
                self._load(source, target_name)
            elif isinstance(source, Path):
                self._load(pd.read_csv(source), target_name)
            else:
                raise ValueError("Either path or data must be specified")
        else:
            raise ValueError("Either path or data must be specified")

    def split(self, split_size: float = 0.2, **kwargs) -> None:
        """Split the data into training and validation sets
        Parameters
        ----------
        split_size (float) : the size of the validation set

        Returns
        -------
        None
        """
        if self.dataframe is None:
            raise ValueError("No data to split")

        train_df, val_df = train_test_split(
            self._dataframe, test_size=split_size, **kwargs
        )
        self.train_data = (
            train_df.drop(self.target_name, axis=1),
            train_df[self.target_name],
        )
        self.val_data = val_df.drop(self.target_name, axis=1), val_df[self.target_name]

    def _load(self, data: pd.DataFrame, target_name: str = "label") -> None:
        """Load the data into the class
        Parameters
        ----------
        data (pd.DataFrame) : the data to load

        Returns
        -------
        None
        """
        self.dataframe = data  # TODO validate data
        self.target_name = target_name
        self.feature_names = list(
            self.dataframe.drop(self.target_name, axis=1).columns
        )  # TODO validate target name

    def pca(self, n_components: int = 2) -> None:
        """Perform PCA on the data
        Parameters
        ----------
        n_components (int) : the number of components to use

        Returns
        -------
        None
        """
        if self.dataframe is not None:
            if not (
                isinstance(self._train_x, pd.DataFrame)
                and isinstance(self._val_x, pd.DataFrame)
            ):
                raise ValueError("No data to perform PCA")
        else:
            raise ValueError("No data to perform PCA")

        if self._pca_object is not None:
            raise ValueError("PCA already performed")

        self._pca_object = PCA(n_components=n_components)
        pca_train_x = self._pca_object.fit_transform(self._train_x)
        # rename columns of PCA dataframe
        self._pca_train_x = pd.DataFrame(
            pca_train_x, columns=[f"PCA_{i}" for i in range(pca_train_x.shape[1])]
        )
        if self._val_x is not None:
            pca_val_x = self._pca_object.transform(self._val_x)
            # rename columns of PCA dataframe
            self._pca_val_x = pd.DataFrame(
                pca_val_x, columns=[f"PCA_{i}" for i in range(pca_val_x.shape[1])]
            )

    @property
    def train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """The training set
        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            the training set

        Raises
        ------
        ValueError
            if the training set has not been loaded
        """
        if self._train_x is None:
            raise ValueError("No training data, perform split on Data!")
        if self._pca_train_x is not None:
            return self._pca_train_x, self._train_y
        return self._train_x, self._train_y

    @train_data.setter
    def train_data(self, data: List[pd.DataFrame]) -> None:
        """Set the training set
        Parameters
        ----------
        data (pd.DataFrame) : the training set

        Returns
        -------
        None
        """
        if self._train_x is not None:
            raise ValueError("Training set already set")
        self._train_x, self._train_y = data

    @property
    def val_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """The validation set
        Parameters
        ----------
        pca (bool) : whether to return the PCA transformed data

        Returns
        -------
        pd.DataFrame
            the validation set

        Raises
        ------
        ValueError
            if the validation set has not been loaded
        """
        # TODO exception handling data loaded
        if self._val_x is None:
            raise ValueError("No training data, perform split on Data!")
        if self._pca_val_x is not None:
            return self._pca_val_x, self._val_y
        return self._val_x, self._val_y

    @val_data.setter
    def val_data(self, data: List[pd.DataFrame]) -> None:
        """
        set the validation set
        Parameters
        ----------
        data (pd.DataFrame) : the validation set

        Returns
        -------
        None
        """
        self._val_x, self._val_y = data

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
        return f"Data(name={self.name!r}, dataframe shape={self._dataframe.shape!r})"
