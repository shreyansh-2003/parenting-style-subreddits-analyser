import numpy as np
from umap import UMAP


class Visualisation:
    def __init__(self):
        pass

    def reduce_dimensionality(self, embeddings: np.ndarray, n_components: int = 2, n_neighbors: int = 15, metric: str = 'cosine'):
        """
        Reduce the dimensionality of a set of embeddings using UMAP.
        """
        # Initialize UMAP model
        umap_model = UMAP(n_components=n_components,
                          n_neighbors=n_neighbors, metric=metric)

        # Fit and transform the embeddings
        umap_embeddings = umap_model.fit_transform(embeddings)

        return umap_embeddings
