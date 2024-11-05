import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                f'Word Similarities in {subreddit} (Top {n_highlight} Terms Highlighted)')
        else:
            ax.set_title(
                f'Word Similarities (Top {n_highlight} Terms Highlighted)')

        # Set axis labels
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')

        return fig, ax

    def plot_tsne_3d(self, embeddings: np.ndarray, posts_df: pd.DataFrame,
                     n_components: int = 3, perplexity: int = 30,
                     subreddit_column: str = 'subreddit',
                     title: str = 't-SNE of Reddit posts'):
        """
        Plot word embeddings in 3D using t-SNE.
        """
        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=n_components,
                    perplexity=perplexity,
                    random_state=42)
        tsne_embeddings = tsne.fit_transform(embeddings)

        # Create 3D plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each subreddit with different color
        for subreddit in posts_df[subreddit_column].unique():
            idx = posts_df[subreddit_column] == subreddit
            ax.scatter(tsne_embeddings[idx, 0],
                       tsne_embeddings[idx, 1],
                       tsne_embeddings[idx, 2],
                       label=subreddit,
                       alpha=0.4,
                       s=15)

        # Set labels and title
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
        ax.set_title(title)
        ax.legend()

        return fig, ax

    def plot_subreddit_vector_space(self, posts_df: pd.DataFrame, embeddings_column: str,
                                    perplexity: int = 2, title=None, labels=None):
        """
        Plot embedding vectors in 2D space using t-SNE.
        """
        # Calculate an average embedding for each subreddit
        subreddit_centroids = np.vstack(posts_df.groupby(
            'subreddit')[embeddings_column].mean().values)

        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    random_state=42)
        tsne_centroids = tsne.fit_transform(subreddit_centroids)

        # Create the plot
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Plot vectors from origin to each point
        colors = plt.cm.Paired(np.linspace(0, 1, len(tsne_centroids)))

        for i, (x, y) in enumerate(tsne_centroids):
            # Plot the vector from origin
            plt.quiver(0, 0, x, y,
                       angles='xy',
                       scale_units='xy',
                       scale=1,
                       color=colors[i],
                       width=0.005,
                       label=labels[i] if labels is not None else f'Vector {i+1}')

        # Style the plot
        plt.grid(True, linestyle='--', alpha=0.7)

        # Set axis limits
        max_val = np.max(np.abs(tsne_centroids)) * 1.1
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)

        # Make the plot square
        ax.set_aspect('equal')

        # Style the axes
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Labels
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(title or "Embedding Vectors in 2D Space")

        plt.legend()

        plt.tight_layout()

        return fig, ax
