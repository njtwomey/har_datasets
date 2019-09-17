from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as pl
import seaborn as sns

from umap import UMAP

from .base import VisualisationBase
from .. import normalise_labels

sns.set_style('darkgrid')
sns.set_context('paper')


def learn_umap(key, label, data):
    umap = Pipeline((
        ('scale', StandardScaler()),
        ('embed', UMAP(n_neighbors=100 // 2, verbose=True)),
    ))
    
    umap.fit(data)
    
    return umap


def embed_umap(key, label, data, model):
    embedding = model.transform(data)
    
    label.track_0[:] = label.track_0.apply(normalise_labels)
    
    fig, ax = pl.subplots(1, 1, figsize=(10, 10))
    
    labels = label.track_0.unique()
    colours = sns.color_palette(n_colors=labels.shape[0])
    for ll, cc in zip(labels, colours):
        if ll == 'other':
            continue
        inds = label.track_0 == ll
        ax.scatter(
            embedding[inds, 0],
            embedding[inds, 1],
            c=cc, label=ll, s=5,
            alpha=0.75
        )
    pl.legend(fontsize='x-large', markerscale=3)
    pl.tight_layout()
    pl.tight_layout()
    
    return fig


class umap_embedding(VisualisationBase):
    def __init__(self, parent):
        super(umap_embedding, self).__init__(
            name=self.__class__.__name__, parent=parent,
        )
        
        label = parent.index['label']
        
        for key, node in parent.outputs.items():
            model = self.outputs.make_output(
                key=key + ('umap',),
                func=learn_umap,
                sources=dict(
                    label=label,
                    data=node
                ),
                backend='none'
            )
            
            self.outputs.add_output(
                key=('viz',) + key,
                func=embed_umap,
                sources=dict(
                    label=label,
                    data=node,
                    model=model
                ),
                backend='png'
            )
