# import python standard libraries
import json
from typing import Optional

# import 3rd party libraries
import dice_ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dice_ml.utils import helpers  # helper functions
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def prep_data(dataframe: pd.DataFrame,
              target: str = None,
              split: bool = None) -> list[pd.DataFrame, pd.DataFrame]:

    """
    separate into labels and training images, 
    and split into train and test sets

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe to split
    target : str
        name of the target column
    split : bool
        split the data into train and test sets

    Returns
    -------
    list[pd.DataFrame, pd.DataFrame]
        train and test sets
    """

    labels = dataframe.loc[:, target]
    input_x = dataframe.iloc[:, 1:] / 255.0
    if split:
        # training and validation split
        train_val_data = train_test_split(input_x, labels,
                                          stratify=labels, random_state=123,
                                          test_size=0.20)
        return train_val_data
    return (input_x, labels)


def train(train_X, train_Y, learner='classifier'):
    """
    train a model
    """

    if learner == 'classifier':
        #perform necessary imports
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(solver='adam',
                            hidden_layer_sizes=(100,),
                            random_state=1,
                            verbose=True)
        best_model = clf.fit(train_X, train_Y)

    return best_model


def evaluate(x_test, y_test, model=None, return_df=None, conf_mat=None, keep_index=None):
    """
    evaluate a model

    Parameters
    ----------
    x_test : pd.DataFrame
        test data
    y_test : pd.Series
        test labels
    model : model
        model to evaluate
    return_df : bool
        return a dataframe of predictions
    conf_mat : bool
        return a confusion matrix

    Returns
    -------
    pd.DataFrame
        predictions
    pd.DataFrame
        confusion matrix
    """

    predictions = model.predict(x_test)
    print(f"{model.score(x_test, y_test) * 100 :.2f} % ")
    if conf_mat:
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        disp.figure_.suptitle("Confusion Matrix")
    if return_df:
        index = y_test.index
        return pd.DataFrame(predictions, columns=['predictions'], index=y_test.index)


def get_query(predictions, true_labels, expected, predicted, count=None, dataframe=None):
    """
    get a query from the dataset
    """

    if dataframe:
        # test_df = pd.DataFrame([predictions, true_labels], index=true_labels.index).transpose()
        test_df = pd.concat([predictions, true_labels], axis=1)
        test_df.columns = ['predictions', 'true_labels']
        condition = f"(predictions != true_labels) & (true_labels == {expected}) & (predictions == {predicted})"
        query_list = test_df.query(condition).index
        return query_list
    else:
        query_list = np.where((predictions != true_labels) & (true_labels == expected) & (predictions == predicted))
    if (count is not None) and (count <= len(query_list[0])):
        return query_list[0][:count]
    else:
        return query_list[0]


def prep_data_for_dice(x_test, y_test):
    """
    prepare data for dice
    """

    # dataframe = pd.concat([x_test, y_test], axis=1,)
    if isinstance(x_test, pd.DataFrame):
        dataframe = x_test.copy()
    else:
        dataframe = pd.DataFrame(x_test)
    dataframe[y_test.name] = y_test.values
    outcome_name = y_test.name
    cont_features = dataframe.drop(outcome_name, axis=1).columns.tolist()
    data_dice = dice_ml.Data(dataframe=dataframe, continuous_features=cont_features, outcome_name=outcome_name)
    return data_dice


def plotter(ax, data, **param_dict):
    """A helper function to make a graph"""
    data = data.reshape((28, 28))
    out = ax.imshow(data, **param_dict)
    return out


def plot_counterfactuals(explainer, pca=None) -> None:
    """
    Plot the query and the resulting counterfactuals
    
    Parameters
    ----------
    explainer : dice_ml.Dice
        explainer object
    pca : sklearn.decomposition.PCA
        pca object
    
    Returns
    -------
    None

    """

    # serialize data from explainer for visualization
    results = json.loads(explainer.to_json())
    query = results['test_data'][0]
    cfs = results['cfs_list'][0]

    # set up plot
    n_cols = len(cfs) + 1
    fig, ax = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2))

    # plot the query
    if pca:
        query_x = pca.inverse_transform(np.array(query[0])[:-1])
    else:
        query_x = np.array(query[0])[:-1]
    plotter(ax[0], query_x)  # exluding the label

    # plot all the counterfactuals
    for idx, img_data in enumerate(cfs, start=1):
        data = np.array(img_data)[:-1]
        if pca:
            data = pca.inverse_transform(data)
        plotter(ax[idx], data,)


def plot_digits(data, pca=None, n_rows: int = 4, n_cols: int = 4):

    fig, axes = plt.subplots(n_rows, n_cols,  # TODO: specify figsize and number of rows and columns
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        try:
            to_draw = data[i]
        except IndexError as e:
            print(f"Warning: number of images less than plot size!")
            break
        if pca:
            to_draw = pca.inverse_transform(to_draw)
        ax.imshow(to_draw.reshape(28, 28),
                  cmap='binary', interpolation='nearest')

    

def plot_difference(data_1, data_2, pca=None, subtract_before=None, return_diff=None, **kwargs):
    """
    plot the difference between two images

    Parameters
    ----------
    data_1 : np.array
        first image
    data_2 : np.array
        second image
    pca : sklearn.decomposition.PCA
        pca object
    subtract_before : bool
        subtract before or after pca
    kwargs : dict
        keyword arguments for plot_digits

    Returns
    -------
    L2 norm of the difference
    difference between the images
    """

    n_rows = 1
    n_cols = 1
    fig, ax = plt.subplots(n_rows, n_cols,  # TODO: specify figsize and number of rows and columns
                           subplot_kw={'xticks': [], 'yticks': []},
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
    # find difference between data_1 and data_2
    if pca:
        if subtract_before:
            to_draw = pca.inverse_transform(data_1 - data_2)
        else:
            to_draw = pca.inverse_transform(data_1) - pca.inverse_transform(data_2)
    else:
        to_draw = data_1 - data_2
    # find difference between two images

    c = ax.imshow(to_draw.reshape(28, 28),
                  cmap='viridis', interpolation='nearest', vmin=0)
    fig.colorbar(c, ax=ax)
    difference = np.linalg.norm(to_draw)
    if return_diff:
        return to_draw, difference
    return difference


def get_PCA_data(
    data: pd.DataFrame,
    n_components: int = None,
    pca: Optional[PCA] = None,
    rename_column: bool = True,
    return_pca: bool = False,
    ) -> list[pd.DataFrame, Optional[PCA]]:
    """
    Get PCA data
    
    Parameters
    ----------
    data : pd.DataFrame
        data to transform
    n_components : int
        number of components to keep
    pca : Optional[PCA]
        PCA object to use
    rename_column : bool
        rename columns to PCA0, PCA1, ...
    return_pca : bool
        return the PCA object

    Returns
    -------
    list[pd.DataFrame, Optional[PCA]]
        transformed data and PCA object
    """
    
    if pca is None:
        pca = PCA(n_components=n_components).fit(data)
    data_pca = pca.transform(data)
    if rename_column:
        # rename columns starting with "feature"
        data_pca = pd.DataFrame(data_pca, columns=[f"PCA_{i}" for i in range(data_pca.shape[1])])
    if return_pca:
        return data_pca, pca
    else:
        return data_pca