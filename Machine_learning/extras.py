import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union


def variance_threshold(df:pd.DataFrame, 
                       threshold: float):
    scaled_df = pd.DataFrame(MinMaxScaler()
             .fit_transform(
                 df.select_dtypes("number")
             ), 
             columns=df.columns)
    summary_df = (scaled_df
     .var()
     .to_frame(name="variance")
     .assign(feature_type=df.dtypes)
     .assign(discard=lambda x: x["variance"] < threshold)
    )
    
    return summary_df


def nunique_threshold(df:pd.DataFrame, 
                       threshold: int):
    scaled_df = pd.DataFrame(MinMaxScaler()
             .fit_transform(
                 df.select_dtypes("number")
             ), 
             columns=df.columns)
    summary_df = (scaled_df
     .apply(lambda x: x.nunique())
     .to_frame(name="nunique")
                  .assign(percent_unique=lambda x: x["nunique"] / df.shape[0] * 100)
     .assign(feature_type=df.dtypes)
     .assign(discard=lambda x: x["nunique"] < threshold)
    )
    
    return summary_df

swell_eda_features_cols = ['MEAN',
 'MAX',
 'MIN',
 'RANGE',
 'KURT',
 'SKEW',
 'MEAN_1ST_GRAD',
 'STD_1ST_GRAD',
 'MEAN_2ND_GRAD',
 'STD_2ND_GRAD',
 'ALSC',
 'INSC',
 'APSC',
 'RMSC',
 'MIN_PEAKS',
 'MAX_PEAKS',
 'STD_PEAKS',
 'MEAN_PEAKS',
 'MIN_ONSET',
 'MAX_ONSET',
 'STD_ONSET',
 'MEAN_ONSET']

swell_eda_target_cols = [
 'condition',
 'Valence',
 'Arousal',
 'Dominance',
 'Stress',
 'MentalEffort',
 'MentalDemand',
 'PhysicalDemand',
 'TemporalDemand',
 'Effort',
 'Performance',
 'Frustration',
 'NasaTLX',
 ]


def plot_confusion_matrix(
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize: tuple = (7, 5),
    cmap="Blues",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
    title:         Title for the heatmap. Default is None.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    _ = sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        _ = plt.ylabel("True label")
        _ = plt.xlabel("Predicted label" + stats_text)
    else:
        _ = plt.xlabel(stats_text)

    if title:
        _ = plt.title(title)

    return fig, ax


def tSNE(
    data: pd.DataFrame,
    n_components: int = 2,
    normalize: bool = True,
    hue: Union[str, None] = None,
    tag: Union[str, None] = None,
    label_fontsize: int = 14,
    figsize: tuple = (11.7, 8.27),
    **kwargs,
):
    r"""Perform t-Distributed Stochastic Neighbor Embedding (t-SNE) Analysis
    More info : https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe which contains some numerical feature
    n_components : int, optional (default: 2)
        Dimension of the embedded space (2D or 3D).
    normalize : bool, optional (default: True)
        Normalize data prior tSNE.
    hue: string, optional
        Grouping variable that will produce points with different colors.
        Can be either categorical or numeric, although color mapping will behave
        differently in latter case.
    tag: string, optional
        Tag each point with the value relative to the corresponding column (only 2D currently)
    label_fontsize: int, optional (default: 14)
        Font size for the `tag`
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    kwargs: key, value pairings
        Additional keyword arguments relative to tSNE() function. Additional info:
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    Returns
    ----------
    fig: matplotlib.pyplot.Figure
        Graph with clusters in embedded space
    Examples
    ----------
    >>> data = sns.load_dataset("mpg")
    >>> from neuropy.cluster_analysis import tSNE
    >>> fig = tSNE(data, n_components=2, hue='origin', tag='name', generate_plot = False)
    """

    # check hue input
    if hue is None:
        warnings.warn(
            "A hue has not been set, hence it shall not be shown in the plot."
        )
    if not isinstance(hue, str):
        raise ValueError("hue input needs to be a string")

    if hue not in data.columns:
        raise ValueError(f"hue='{hue}' is not contained in dataframe")

    # check tag input
    if tag is not None:
        if not isinstance(tag, str):
            raise ValueError("tag needs to be str type")
        if tag not in data.columns:
            raise ValueError(f"tag='{tag}' is not contained in dataframe")
    else:
        pass

    # t-SNE takes into account only numerical feature
    data_num = data.select_dtypes(include="number")
    if hue in data_num.columns:
        data_num = data_num.drop(hue, axis=1)
    data_obj = data.select_dtypes(exclude="number")
    if hue and hue not in data_obj.columns:
        # Add hue column if it is numerical
        data_obj = pd.concat([data_obj, data[[hue]]], axis=1)

    # remove any row with NaNs and normalize data with z-score
    data_num = data_num.dropna(axis="index", how="any")

    if normalize:
        # get z-score to treat different dimensions with equal importance
        data_num = StandardScaler().fit_transform(data_num)

    # Apply t-SNE to normalized_movements: normalized_data
    tsne_features = TSNE(n_components=n_components, **kwargs).fit_transform(data_num)

    # show t-SNE cluster
    if n_components == 2:
        # combine tsne feature with categorical and/or object variables
        df_tsne = pd.DataFrame(data=tsne_features, columns=["t-SNE (x)", "t-SNE (y)"])
        df_tsne = pd.concat([df_tsne, data_obj], axis=1)
        # plot 2D
        fig, ax = plt.subplots(figsize=figsize)
        _ = plt.title("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        _ = sns.scatterplot(
            x="t-SNE (x)",
            y="t-SNE (y)",
            hue=hue,
            legend="full",
            data=df_tsne,
            alpha=0.8,
        )

    elif n_components == 3:
        # combine tsne feature with categorical and/or object variables
        df_tsne = pd.DataFrame(
            data=tsne_features, columns=["t-SNE (x)", "t-SNE (y)", "t-SNE (z)"]
        )
        df_tsne = pd.concat([df_tsne, data_obj], axis=1)
        # plot 3D
        fig = plt.figure(figsize=figsize)
        _ = plt.title("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        ax = fig.add_subplot(111, projection="3d")
        if hue:
            i = ax.scatter(
                df_tsne["t-SNE (x)"],
                df_tsne["t-SNE (y)"],
                df_tsne["t-SNE (z)"],
                c=df_tsne[hue],
                cmap="tab10",
                s=60,
                alpha=0.8,
            )
            fig.colorbar(i)
        else:
            ax.scatter(
                df_tsne["t-SNE (x)"],
                df_tsne["t-SNE (y)"],
                df_tsne["t-SNE (z)"],
                c="#75bbfd",
                s=60,
                alpha=0.8,
            )
        _ = ax.view_init(30, 185)

    else:
        raise ValueError("n_components can be either 2 or 3")

    # tag each point
    if tag is not None and n_components == 2:
        for x, y, tag in zip(df_tsne["t-SNE (x)"], df_tsne["t-SNE (y)"], df_tsne[tag]):
            plt.annotate(tag, (x, y), fontsize=label_fontsize, alpha=0.75)

    return fig, ax