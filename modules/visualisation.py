import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP


class Visualisation:
    def __init__(self):
        pass

    def reduce_dimensionality(self, embeddings: np.ndarray, n_components: int = 2,
                              n_neighbors: int = 15, metric: str = 'cosine'):
        """
        Reduce the dimensionality of a set of embeddings using UMAP.
        """
        # Initialize UMAP model
        umap_model = UMAP(n_components=n_components,
                          n_neighbors=n_neighbors,
                          metric=metric)

        # Fit and transform the embeddings
        umap_embeddings = umap_model.fit_transform(embeddings)

        return umap_embeddings

    def plot_3d_umap(self, posts_df: pd.DataFrame, umap_results: np.ndarray,
                     subreddit_column: str = 'subreddit', title: str = 'UMAP of Reddit posts',
                     figsize: tuple = (8, 8)):
        """
        Plot a 3D UMAP visualization of Reddit posts.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for subreddit in posts_df['subreddit'].unique():
            idx = posts_df[subreddit_column] == subreddit
            ax.scatter(umap_results[idx, 0],
                       umap_results[idx, 1],
                       umap_results[idx, 2],
                       label=subreddit,
                       alpha=0.4,
                       s=15)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        ax.set_title(title)
        ax.legend()

        return fig, ax
