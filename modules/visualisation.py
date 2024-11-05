import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.manifold import TSNE


class Visualisation:
    def __init__(self):
        pass

    def plot_word_similarities_tsne_3d(self,
                                       tfidf_matrix: np.ndarray,
                                       feature_names: list,
                                       n_highlight: int = 5,
                                       subreddit: str = None):
        """
        Plot word similarities using t-SNE in 3D with all terms but highlighting top N.
        """
        # Get vectors for all terms
        term_vectors = tfidf_matrix.T.toarray()

        # Identify top terms
        mean_tfidf = tfidf_matrix.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-n_highlight:][::-1]
        top_terms = feature_names[top_indices]

        # Calculate t-SNE for all terms with 3 components
        tsne = TSNE(n_components=3,
                    perplexity=min(30, len(feature_names) - 1),
                    random_state=42)
        coords = tsne.fit_transform(term_vectors)

        # Create 3D plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all points in light grey
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c='lightgrey', alpha=0.5, zorder=1)

        # Highlight top terms
        ax.scatter(coords[top_indices, 0],
                   coords[top_indices, 1],
                   coords[top_indices, 2],
                   c='red', alpha=0.8, s=150, zorder=2)

        # Calculate offset for labels to prevent overlap with points
        offset = np.array([0.1, 0.1, 0.1])

        # Add labels for top terms with black background and white text
        for i, term in enumerate(top_terms):
            point_coords = coords[top_indices[i]]
            label_coords = point_coords + offset

            ax.text(label_coords[0],
                    label_coords[1],
                    label_coords[2],
                    term,
                    color='white',
                    fontsize=14,
                    bbox=dict(facecolor='black',
                              edgecolor='none',
                              alpha=0.7,
                              pad=5),
                    ha='center',
                    va='center',
                    zorder=4)

        if subreddit:
            ax.set_title(
                f'3D Word Similarities in {subreddit} (Top {n_highlight} Terms Highlighted)')
        else:
            ax.set_title(
                f'3D Word Similarities (Top {n_highlight} Terms Highlighted)')

        # Set axis labels
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')

        # Add padding
        ax.dist = 12

        return fig, ax

    def reduce_dimensionality_umap(self, embeddings: np.ndarray, n_components: int = 2,
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

    def plot_3d_umap(self, posts_df: pd.DataFrame,
                     umap_results: np.ndarray,
                     subreddit_column: str = 'subreddit',
                     title: str = 'UMAP of Reddit posts',
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
