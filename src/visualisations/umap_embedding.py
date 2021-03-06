import matplotlib.pyplot as pl
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.keys import Key
from src.utils.label_helpers import normalise_labels
from src.visualisations.base import VisualisationBase

sns.set_style("darkgrid")
sns.set_context("paper")


def learn_umap(key, label, data):
    umap = Pipeline([("scale", StandardScaler()), ("embed", UMAP(n_neighbors=50, verbose=True))])

    umap.fit(data)

    return umap


def embed_umap(key, label, data, model):
    embedding = model.transform(data)

    # Need to re-label with a new dataframe since the categories in the normalised label
    # set are different to those in the full set.
    label = pd.DataFrame(label.target.apply(normalise_labels)).astype("category")

    fig, ax = pl.subplots(1, 1, figsize=(10, 10))

    labels = label.target.values
    unique_labels = label.target.unique()
    colours = sns.color_palette(n_colors=unique_labels.shape[0])
    for ll, cc in zip(unique_labels, colours):
        if ll == "other":
            continue
        inds = labels == ll
        ax.scatter(embedding[inds, 0], embedding[inds, 1], c=cc, label=ll, s=5, alpha=0.75)
    pl.legend(fontsize="x-large", markerscale=3)
    pl.tight_layout()

    return fig


class umap_embedding(VisualisationBase):
    def __init__(self, parent, task):
        super(umap_embedding, self).__init__(
            name=self.__class__.__name__, parent=parent,
        )

        label = task.index["target"]

        for key, node in parent.outputs.items():
            model = self.outputs.make_output(
                key=tuple(key) + ("umap",),
                func=learn_umap,
                backend="none",
                kwargs=dict(label=label, data=node),
            )

            self.outputs.add_output(
                key=tuple(key) + ("umap", "viz"),
                func=embed_umap,
                backend="png",
                kwargs=dict(data=node, model=model, label=label),
            ).evaluate()
