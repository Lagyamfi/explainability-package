# import python standard libraries
import json
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

# import 3rd party libraries
import dice_ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import torch
import umap
import wandb
from dice_ml.utils import helpers  # helper functions
from scipy.spatial.distance import cosine, mahalanobis
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

MAIN_DIR = Path.cwd().parent
IMG_SAVE_DIR = MAIN_DIR / "results" / "plots"

ALGORITHMS = [
    "Random Forest",
    "Logistic Regression",
    "MLP (sklearn)",
    "Decision Tree",
    "KNN",
    "MLP(Pytorch - Gradient CFs",
    "MLP(Pytorch - Random CFs",
]


def prep_data(
    dataframe: pd.DataFrame,
    target: str = "",
    split: bool = False,
    reshape: bool = False,
) -> List[Any]:

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
    input_x = dataframe.drop(target, axis=1) / 255.0
    if split:
        # training and validation split
        train_val_data = train_test_split(
            input_x, labels, stratify=labels, random_state=123, test_size=0.20
        )
        if reshape:
            train_val_data = [
                data.values.reshape(-1, 28, 28) for data in train_val_data
            ]
        return train_val_data

    if reshape:
        input_x = input_x.values.reshape(-1, 28, 28)

    return [input_x, labels]


def train(train_X, train_Y, learner="classifier"):
    """
    train a model
    """

    if learner == "classifier":
        # perform necessary imports
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(
            solver="adam", hidden_layer_sizes=(100,), random_state=1, verbose=True
        )
        best_model = clf.fit(train_X, train_Y)

    return best_model


def evaluate(
    x_test, y_test, model=None, return_df=None, conf_mat=None, keep_index=None
):
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
        return pd.DataFrame(predictions, columns=["predictions"], index=y_test.index)


def get_query(
    predictions, true_labels, expected, predicted, count=None, dataframe=None
):
    """
    get a query from the dataset
    """

    if dataframe:
        # test_df = pd.DataFrame([predictions, true_labels], index=true_labels.index).transpose()
        test_df = pd.concat([predictions, true_labels], axis=1)
        test_df.columns = ["predictions", "true_labels"]
        condition = f"(predictions != true_labels) & (true_labels == {expected}) & (predictions == {predicted})"
        query_list = test_df.query(condition).index
        return query_list
    else:
        query_list = np.where(
            (predictions != true_labels)
            & (true_labels == expected)
            & (predictions == predicted)
        )
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
    data_dice = dice_ml.Data(
        dataframe=dataframe,
        continuous_features=cont_features,
        outcome_name=outcome_name,
    )
    return data_dice


def plotter(ax, data, **param_dict):
    """A helper function to make a graph"""
    data = data.reshape((28, 28))
    out = ax.imshow(data, **param_dict)
    return out


def plot_counterfactuals(explainer, pca=None, id=0, **kwargs) -> None:
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
    query = results["test_data"][id]
    cfs = results["cfs_list"][id]

    # set up plot
    n_cols = len(cfs) + 2
    fig, ax = plt.subplots(1, n_cols, figsize=(3 * n_cols, 2))

    # plot the query and average image
    query_x = np.array(query[0])[:-1]
    avg_cf = np.array(cfs).mean(axis=0)[:-1]
    if pca:
        query_x = pca.inverse_transform(query_x)
        avg_cf = pca.inverse_transform(avg_cf)

    plotter(ax[0], query_x, **kwargs)  # exluding the label
    ax[0].set_title("query")
    plotter(ax[1], avg_cf, **kwargs)
    norm_avg = np.linalg.norm((query_x - avg_cf))
    ax[1].set_title(f"avg CF: {norm_avg:.1f}")

    # plot all the counterfactuals
    for idx, img_data in enumerate(
        cfs,
        start=2,
    ):
        data = np.array(img_data)[:-1]
        if pca:
            data = pca.inverse_transform(data)
        norm = np.linalg.norm((data - query_x))
        plotter(
            ax[idx],
            data,
            **kwargs,
        )
        ax[idx].set_title(f"norm: {norm:.1f}")


def plot_digits(data, pca=None, n_rows: int = 4, n_cols: int = 4):

    fig, axes = plt.subplots(
        n_rows,
        n_cols,  # TODO: specify figsize and number of rows and columns
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for i, ax in enumerate(axes.flat):
        try:
            to_draw = data[i]
        except IndexError as e:
            print("Warning: number of images less than plot size!")
            break
        if pca:
            to_draw = pca.inverse_transform(to_draw)
        ax.imshow(to_draw.reshape(28, 28), cmap="gray", vmin=0, vmax=255)


def plot_difference(
    data_1, data_2, pca=None, subtract_before=None, return_diff=None, **kwargs
):
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
    fig, ax = plt.subplots(
        n_rows,
        n_cols,  # TODO: specify figsize and number of rows and columns
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    # find difference between data_1 and data_2
    if pca:
        if subtract_before:
            to_draw = pca.inverse_transform(data_1 - data_2)
        else:
            to_draw = pca.inverse_transform(data_1) - pca.inverse_transform(data_2)
    else:
        to_draw = data_1 - data_2
    # find difference between two images

    c = ax.imshow(to_draw.reshape(28, 28), **kwargs)
    fig.colorbar(c, ax=ax)
    difference = np.linalg.norm(to_draw)
    if return_diff:
        return to_draw, difference
    return difference


def get_PCA_data(
    data: pd.DataFrame,
    n_components: int = 2,
    pca: Optional[PCA] = None,
    rename_column: bool = True,
    return_pca: bool = False,
) -> Tuple[pd.DataFrame, Optional[PCA]]:
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
        data_pca = pd.DataFrame(
            data_pca, columns=[f"PCA_{i}" for i in range(data_pca.shape[1])]
        )
    if return_pca:
        return data_pca, pca
    else:
        return data_pca


def conf_matrix(function: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to print a confusion matrix
    """

    @wraps(function)
    def wrapper(data, labels):
        eval_results = function(data, labels)
        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            labels, eval_results["predictions"]
        )
        disp.figure_.suptitle("Confusion Matrix")

    return wrapper


def data_umap(path, id, is_torch=False, return_exp=False):
    with open(path / "exp_0_full.pkl", "rb") as f:
        exp_4 = pickle.load(f)

    with open(path / "exp_1_full.pkl", "rb") as f:
        exp_9 = pickle.load(f)
    if is_torch:
        ml_model = torch.load(path / "model.pt")
    else:
        ml_model = pickle.load(open(path / "model.pkl", "rb"))

    data_cf_4 = [
        data.final_cfs_df.drop("label", axis=1).values
        for data in exp_4.cf_examples_list
    ]
    data_instance_4 = [
        data.test_instance_df.drop("label", axis=1).values
        for data in exp_4.cf_examples_list
    ]
    df_4 = pd.DataFrame([data.squeeze() for data in data_instance_4])
    df_4["label"] = 4
    df_4["preds"] = ml_model.predict(df_4.drop("label", axis=1))
    # change from (0 and 1) to (4 and 9)
    df_4["preds"] = df_4["preds"].apply(lambda x: 4 if x == 0 else 9)
    # df_4['norms'] = [np.linalg.norm(data_instance_4[i] - data_cf_4[i].mean(axis=0), 2) for i in range(len(data_instance_4))]
    difference = [
        data_instance_4[i] - data_cf_4[i].mean(axis=0)
        for i in range(len(data_instance_4))
    ]
    df_4["norms"] = [np.linalg.norm(diff, 2) for diff in difference]
    df_4["hamming"] = [np.linalg.norm(diff, 1) for diff in difference]
    df_4["cosine"] = [
        cosine(data_instance_4[i].squeeze(), data_cf_4[i].mean(axis=0).squeeze())
        for i in range(len(data_instance_4))
    ]

    data_cf_9 = [
        data.final_cfs_df.drop("label", axis=1).values
        for data in exp_9.cf_examples_list
    ]
    data_instance_9 = [
        data.test_instance_df.drop("label", axis=1).values
        for data in exp_9.cf_examples_list
    ]
    df_9 = pd.DataFrame([data.squeeze() for data in data_instance_9])
    df_9["label"] = 9
    df_9["preds"] = ml_model.predict(df_9.drop("label", axis=1))
    df_9["preds"] = df_9["preds"].apply(lambda x: 9 if x == 1 else 4)
    # df_9['norms'] = [np.linalg.norm(data_instance_9[i] - data_cf_9[i].mean(axis=0), 2) for i in range(len(data_instance_9))]
    difference = [
        data_instance_9[i] - data_cf_9[i].mean(axis=0)
        for i in range(len(data_instance_9))
    ]
    df_9["norms"] = [np.linalg.norm(diff, 2) for diff in difference]
    df_9["hamming"] = [np.linalg.norm(diff, 1) for diff in difference]
    df_9["cosine"] = [
        cosine(data_instance_9[i].squeeze(), data_cf_9[i].mean(axis=0).squeeze())
        for i in range(len(data_instance_9))
    ]

    df = pd.concat([df_4, df_9])

    if return_exp:
        return df, ml_model, exp_4, exp_9

    return df, ml_model


def get_embedding(
    dataframe,
    n_cols=None,
    parametric=None,
    pca=None,
    log=None,
    n_neighbors=4,
    min_dist=0.2,
    verbose=False,
    **kwargs,
):
    assert n_cols is not None, "Must specify n_cols"
    data = dataframe.iloc[:, :n_cols]

    # log to wandb
    if log:
        n_neighbors = wandb.config.n_neighbors
        min_dist = wandb.config.min_dist
        metric = wandb.config.dist_metric

    # parametric
    if parametric:
        embedder = umap.ParametricUMAP(n_epochs=50, verbose=verbose, **kwargs).fit(data)
        embedding = embedder.transform(data)
    # pca
    else:
        embedder = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
            **kwargs,
        ).fit(data)
        embedding = embedder.transform(data)

    df_new = pd.DataFrame(embedding, columns=["x", "y"])
    df_new["norms"] = dataframe["norms"].values
    df_new["label"] = dataframe["label"].values
    df_new["preds"] = dataframe["preds"].values

    return embedder, df_new


def draw_umap(
    dataframe,
    embedder,
    log=None,
    hist_func="avg",
    nbins=5,
    histnorm="percent",
    distance="norms",
    **kwargs,
):
    norms = dataframe[distance]
    labels = dataframe["label"]
    preds = dataframe["preds"]
    diff = dataframe[dataframe["label"] != dataframe["preds"]]

    SYMBOLS = {
        "4": ["red", "star"],
        "9": [
            "green",
            "diamond",
        ],
    }

    if log:
        hist_func = wandb.config.hist_func
        n_neighbors = wandb.config.n_neighbors
        metric = wandb.config.dist_metric
        min_dist = wandb.config.min_distdata_exp_4
        algorithm = wandb.config.algorithm
    else:
        hist_func = hist_func
        n_neighbors = embedder.n_neighbors
        metric = embedder.metric
        min_dist = embedder.min_dist
        algorithm = kwargs.get("algorithm", None)

    customdata = pd.DataFrame(
        {"norms": norms, "labels": labels, "preds": preds, "index": dataframe.index}
    )

    fig = go.Figure()

    # draw the contour plot with norms as the z axis
    fig.add_trace(
        go.Histogram2dContour(
            x=dataframe.x,
            y=dataframe.y,
            z=norms,
            nbinsx=nbins,
            nbinsy=nbins,
            histfunc=hist_func,
            histnorm=histnorm,
            colorscale="jet",
            name="norms",
            colorbar=dict(title="norms"),
            hoverinfo="skip",
            contours=dict(showlabels=True),
        )
    )

    if not kwargs.get("only_mis", False):
        # draw the scatter plot with labels as the color
        for label in dataframe.label.unique():
            fig.add_trace(
                go.Scatter(
                    x=dataframe[dataframe.label == label].x,
                    y=dataframe[dataframe.label == label].y,
                    mode="markers",
                    marker=dict(
                        size=5,
                        symbol=SYMBOLS[str(label)][1],
                        color=SYMBOLS[str(label)][0],
                    ),
                    name=str(label),
                    customdata=customdata[customdata["labels"] == label],
                    hovertemplate="<b> norm: %{customdata[0]:.3f} <br>class: %{customdata[1]}<br>preds: %{customdata[2]} <br>index: %{customdata[3]}",
                )
            )

    # draw the misclassified points
    fig.add_trace(
        go.Scatter(
            x=diff.x,
            y=diff.y,
            mode="markers",
            marker=dict(color="red", size=10, symbol="circle-open"),
            name="Misclassified",
            # customdata=diff,
            customdata=pd.DataFrame(
                {
                    "norms": diff.norms,
                    "labels": diff.label,
                    "preds": diff.preds,
                    "index": diff.index,
                }
            ),
            hovertemplate="<b> norm: %{customdata[0]:.3f} <br>class: %{customdata[1]}<br>preds: %{customdata[2]} <br>index: %{customdata[3]}",
        )
    )

    fig.update_layout(
        title=f"{n_neighbors=}, {metric=}, {hist_func=}, {algorithm=!r}",
        xaxis=dict(ticks="", showgrid=True, zeroline=True, nticks=10),
        yaxis=dict(ticks="", showgrid=False, zeroline=True, nticks=10),
        autosize=True,
        height=600,
        width=700,
        hovermode="closest",
        margin_pad=5,
        margin=dict(l=5, r=5, pad=5),
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    return fig


def draw_and_save(data_id, embedder, plot_title="test", **kwargs):
    embedding = eval(f"embedding_{data_id}")
    kwargs.update({"algorithm": ALGORITHMS[data_id - 1]})
    f = draw_umap(embedding, embedder, **kwargs)
    IMG_DIR = str(IMG_SAVE_DIR) + f"/{plot_title}"
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
    filename = kwargs.get("algorithm", None)
    if kwargs.get("only_mis", False):
        filename = f"mis_{filename}"
    pio.write_image(f, f"{IMG_DIR}/{filename}.png", format="png")


def get_embedding_2(
    dataframe,
    n_cols=None,
    parametric=None,
    pca=None,
    log=None,
    n_neighbors=4,
    min_dist=0.2,
    verbose=False,
    only_emb=False,
    **kwargs,
):
    assert n_cols is not None, "Must specify n_cols"
    data = dataframe.iloc[:, :n_cols]

    # log to wandb
    if log:
        n_neighbors = wandb.config.n_neighbors
        min_dist = wandb.config.min_dist
        metric = wandb.config.dist_metric

    # parametric
    if parametric:
        embedder = get_embedder(
            data, parametric=True, n_epochs=50, verbose=verbose, **kwargs
        )
        embedding = embedder.transform(data)

    # pca
    else:
        embedder = get_embedder(
            data,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
            **kwargs,
        )
        embedding = embedder.transform(data)

    if only_emb:
        return embedding, embedder

    df_new = update_embedding(embedding, dataframe)

    return embedder, df_new


def get_embedder(data, parametric=None, **params):
    if parametric:
        embedder = umap.ParametricUMAP(**params).fit(data)
    else:
        embedder = umap.UMAP(**params).fit(data)
    return embedder


def update_embedding(embedding, dataframe, embedder=None, n_cols=None):
    if embedder:
        assert n_cols is not None, "Must specify n_cols"
        embedding = embedder.transform(dataframe.iloc[:, :n_cols])
    df_new = pd.DataFrame(embedding, columns=["x", "y"])
    df_new["norms"] = dataframe["norms"].values
    df_new["hamming"] = dataframe["hamming"].values
    df_new["cosine"] = dataframe["cosine"].values
    df_new["correlation"] = dataframe["correlation"].values
    df_new["label"] = dataframe["label"].values
    df_new["preds"] = dataframe["preds"].values

    return df_new


def get_cfs(exp, idx, embedder=None):
    cfs = [
        data.final_cfs_df.drop("label", axis=1).values for data in exp.cf_examples_list
    ][idx]
    query = [
        data.test_instance_df.drop("label", axis=1).values
        for data in exp.cf_examples_list
    ][idx]
    if embedder:
        # data = [np.expand_dims(cf, 0) for cf in cfs]
        embedding_cf = embedder.transform(cfs)
        embedding_query = embedder.transform(query)
    return embedding_cf, embedding_query


# remove other traces and draw CFs of idx
def draw_cfs(fig, idx, embedder):
    figure = go.Figure(fig)
    # get CFs of idx
    cfs, query = get_cfs(cf_exp_1_4, idx, embedder=embedder)
    # remove other traces
    figure.update_traces(visible="legendonly", selector=dict(name="Misclassified"))
    figure.update_traces(visible="legendonly", selector=dict(name="4"))
    figure.update_traces(visible="legendonly", selector=dict(name="9"))
    # figure.update_traces(visible='legendonly', selector=dict(name="CFs"))
    figure.add_trace(
        go.Scatter(
            x=cfs[:, 0],
            y=cfs[:, 1],
            mode="markers",
            marker=dict(color="black", size=10, symbol="circle"),
            name="CFs",
            hovertemplate="<b> norm: %{customdata[0]:.3f} <br>class: %{customdata[1]}<br>preds: %{customdata[2]} <br>index: %{customdata[3]}",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=query[:, 0],
            y=query[:, 1],
            mode="markers",
            marker=dict(color="red", size=10, symbol="circle"),
            name="Query",
            hovertemplate="<b> norm: %{customdata[0]:.3f} <br>class: %{customdata[1]}<br>preds: %{customdata[2]} <br>index: %{customdata[3]}",
        )
    )

    return figure
