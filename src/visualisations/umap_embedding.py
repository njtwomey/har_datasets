import matplotlib.pyplot as pl
import pandas as pd
import seaborn as sns
from mldb import NodeWrapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src import ExecutionGraph

# from src.utils.label_helpers import normalise_labels

sns.set_style("darkgrid")
sns.set_context("paper")


def learn_umap(data):
    umap = Pipeline([("scale", StandardScaler()), ("embed", UMAP(n_neighbors=50, verbose=True))])
    umap.fit(data)
    return umap


def embed_umap(label, data, model):
    embedding = model.transform(data)

    # Need to re-label with a new dataframe since the categories in the normalised label
    # set are different to those in the full set.
    # label = pd.DataFrame(label.target.apply(normalise_labels)).astype("category")
    label = pd.DataFrame(label["target"]).astype("category")

    fig, ax = pl.subplots(1, 1, figsize=(10, 10))

    labels = label.target.values
    unique_labels = label.target.unique()
    colours = sns.color_palette(n_colors=unique_labels.shape[0])
    for ll, cc in zip(unique_labels, colours):
        if ll == "other":
            continue
        inds = labels == ll
        ax.scatter(embedding[inds, 0], embedding[inds, 1], color=cc, label=ll, s=5, alpha=0.75)
    pl.legend(fontsize="x-large", markerscale=3)
    pl.tight_layout()

    return fig


def umap_embedding(node: NodeWrapper, task_name):
    parent: ExecutionGraph = node.graph
    umap_model = parent.instantiate_orphan_node(func=learn_umap, backend="none", kwargs=dict(data=node),)
    parent.instantiate_node(
        key=f"{parent.identifier.name}-umap",
        func=embed_umap,
        backend="png",
        kwargs=dict(data=node, model=umap_model, label=parent[task_name]),
    )
