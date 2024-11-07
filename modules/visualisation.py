import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from modules.nlp import NLP

nlp = NLP()


class Visualisation:
    def __init__(self, fig_size: tuple = (12, 6)):
        self.fig_size = fig_size

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
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection='3d')

        # Plot all points in light grey
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c='lightgrey', alpha=0.5, zorder=1)

        # Highlight top terms
        ax.scatter(coords[top_indices, 0],
                   coords[top_indices, 1],
                   coords[top_indices, 2],
                   c='red', alpha=0.8, s=100, zorder=2)

        # Calculate offset for labels to prevent overlap with points
        offset = np.ptp(coords, axis=0) * 0.05  # 5% of range

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

        # Set axis limits to ignore outliers
        for dim in range(3):
            dim_coords = coords[:, dim]
            q1 = np.percentile(dim_coords, 25)
            q3 = np.percentile(dim_coords, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            if dim == 0:
                ax.set_xlim(lower, upper)
            elif dim == 1:
                ax.set_ylim(lower, upper)
            else:
                ax.set_zlim(lower, upper)

        if subreddit:
            ax.set_title(
                f'Word Similarities in {subreddit} (Top {n_highlight} Terms Highlighted)')
        else:
            ax.set_title(
                f'Word Similarities (Top {n_highlight} Terms Highlighted)')

        return fig, ax

    def plot_tsne_3d(self, posts_df: pd.DataFrame,
                     embeddings_column: str = 'embeddings',
                     subreddit_column: str = 'subreddit',
                     title: str = 't-SNE of Reddit posts',
                     perplexity: int = 30):
        """
        Plot word embeddings in 3D using t-SNE.
        """
        # Convert embeddings to array
        embeddings = np.vstack(posts_df[embeddings_column].values)

        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=3,
                    perplexity=perplexity,
                    random_state=42)
        tsne_embeddings = tsne.fit_transform(embeddings)

        # Create 3D plot
        fig = plt.figure(figsize=self.fig_size)
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

        # Set title
        ax.set_title(title)
        ax.legend()

        return fig, ax

    def plot_subreddit_vector_space(self, posts_df: pd.DataFrame,
                                    embeddings_column: str = 'embeddings',
                                    perplexity: int = 2,
                                    title=None, labels=None):
        """
        Plot embedding vectors using t-SNE.
        """
        # Calculate a centroid for each subreddit
        centroid_matrix = nlp.calculate_centroids(posts_df, embeddings_column)

        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    random_state=42)
        tsne_centroids = tsne.fit_transform(centroid_matrix)

        # Create plot
        fig, ax = plt.subplots(figsize=self.fig_size)

        # Plot vectors from origin to each point
        colors = plt.cm.Paired(np.linspace(0, 1, len(tsne_centroids)))

        for i, (x, y) in enumerate(tsne_centroids):
            # Plot the vector from origin to point
            ax.quiver(0, 0,        # Start at origin
                      x, y,         # Vector components
                      color=colors[i],
                      angles='xy',
                      scale_units='xy',
                      scale=1,
                      label=labels[i] if labels is not None else f'Vector {i+1}')

        # Style the plot
        ax.grid(True, linestyle='--', alpha=0.3)

        # Set axis limits to account for both positive and negative values
        max_val = np.max(np.abs(tsne_centroids)) * 1.05

        # # Set the axis limits
        plt.xlim(0, max_val)
        plt.ylim(-max_val, 0)

        # # Move the y-axis to the right
        # plt.gca().yaxis.tick_right()
        # plt.gca().yaxis.set_label_position("right")

        # Labels
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.title(title or "Embedding Vectors in 2D Space")

        # Add legend
        plt.legend()

        # Ensure equal aspect ratio for proper vector visualization
        ax.set_aspect('equal')

        return fig, ax

    def plot_subreddit_vector_space_3d(self, posts_df: pd.DataFrame,
                                       embeddings_column: str = 'embeddings',
                                       perplexity: int = 2,
                                       title=None,
                                       labels=None):
        """
        Plot embedding vectors in 3D space using t-SNE.
        """
        # Calculate a centroid for each subreddit
        centroid_matrix = nlp.calculate_centroids(posts_df, embeddings_column)

        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=3,
                    perplexity=perplexity,
                    random_state=42)
        tsne_centroids = tsne.fit_transform(centroid_matrix)

        # Create the 3D plot
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection='3d')

        # Plot vectors from origin to each point
        colors = plt.cm.Paired(np.linspace(0, 1, len(tsne_centroids)))

        for i, (x, y, z) in enumerate(tsne_centroids):
            # Plot the vector from origin to point
            ax.quiver(0, 0, 0,  # Start at origin
                      x, y, z,   # Vector components
                      color=colors[i],
                      arrow_length_ratio=0.1,
                      label=labels[i] if labels is not None else f'Vector {i+1}')

        # Style the plot
        ax.grid(True, linestyle='--', alpha=0.3)

        # Set axis limits
        max_val = np.max(np.abs(tsne_centroids)) * 1.3
        ax.set_xlim([0, max_val])
        ax.set_ylim([0, max_val])
        ax.set_zlim([0, max_val])

        # Labels
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
        plt.title(title or "Embedding Vectors in 3D Space")

        # Add legend
        plt.legend()

        # Adjust view angle
        ax.view_init(elev=20, azim=45)

        return fig, ax
