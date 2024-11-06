import re
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)


class NLP:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))

    def tokenize_text(self, text: str) -> list:
        """
        Clean and normalize text using NLTK.
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string.")

        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]+', ' ', text)
        # Remove single characters
        text = re.sub(r'\b\w\b', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stopwords
        filtered_tokens = [
            word for word in tokens if word not in self.stop_words]

        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Combine tokens into a single string
        token_string = ' '.join(lemmas)

        # Return the tokenized text
        if len(token_string) == 0:
            return None

        return token_string

    def generate_tfidf_matrix(self, texts: list, max_terms: int = 3000, min_doc_freq: int = 2):
        """
        Generate TF-IDF matrix from a list of tokenized texts.
        """
        if not isinstance(texts, list):
            raise ValueError("Texts must be a list of strings.")

        stop_words = list(set(self.stop_words))
        vectorizer = TfidfVectorizer(max_features=max_terms,
                                     min_df=min_doc_freq,
                                     stop_words=stop_words)

        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        return tfidf_matrix, feature_names

    def generate_embeddings(self, tokens: list, word_vectors):
        """
        Generate an average embedding for each document in a corpus based on its token embeddings.
        """
        if not isinstance(tokens, list):
            raise ValueError("Tokens must be a list of strings.")

        # Get embedding dimension from word vectors
        embedding_dim = word_vectors.vector_size

        # Initialize an array to store document embeddings
        num_docs = len(tokens)
        average_embeddings = np.zeros((num_docs, embedding_dim))

        # For each document
        for doc_idx in tqdm(range(num_docs), desc="Generating embeddings"):
            # Get all tokens for the current document
            doc_tokens = tokens[doc_idx].split()

            # Filter to only tokens that exist in the word vectors
            valid_tokens = [
                token for token in doc_tokens if token in word_vectors]

            # Skip documents with no valid tokens
            if not valid_tokens:
                continue

            # Get embeddings for all valid tokens
            token_embeddings = np.array(
                [word_vectors[token] for token in valid_tokens]
            )

            # Compute the document's average embedding
            average_embeddings[doc_idx] = np.mean(token_embeddings, axis=0)

        return average_embeddings

    def generate_tfidf_weighted_embeddings(self, tokens: list, word_vectors,
                                           feature_names: np.ndarray, tfidf_matrix: np.ndarray):
        """
        Generate embeddings weighted by tokens' TF-IDF scores for each document in the corpus.
        """
        if not isinstance(tokens, list):
            raise ValueError("Tokens must be a list of strings.")

        if not isinstance(feature_names, np.ndarray):
            raise ValueError("Feature names must be a Numpy array.")

        # Convert feature names to a list
        feature_names = feature_names.tolist()

        # Get embeding dimension from word vectors
        embedding_dim = word_vectors.vector_size

        # Initialize an array to store document embeddings
        num_docs = len(tokens)
        weighted_embeddings = np.zeros((num_docs, embedding_dim))

        # For each document
        for doc_idx in tqdm(range(num_docs), desc="Generating weighted embeddings"):
            # Get the document's TF-IDF scores
            doc_tfidf = tfidf_matrix[doc_idx].toarray().flatten()

            # Get all tokens for the current document
            doc_tokens = tokens[doc_idx].split()

            # # Filter to only tokens that exist in both word vectors and feature names
            valid_tokens = [
                token for token in doc_tokens
                if token in word_vectors and token in feature_names
            ]

            # Skip documents with no valid tokens
            if not valid_tokens:
                continue

            # Get indices and TF-IDF scores for valid tokens
            token_indices = [feature_names.index(
                token) for token in valid_tokens]
            token_tfidf_scores = doc_tfidf[token_indices]

            # Filter to tokens with positive TF-IDF scores
            positive_mask = token_tfidf_scores > 0
            if not np.any(positive_mask):
                continue

            valid_tokens = [token for token, mask in zip(
                valid_tokens, positive_mask) if mask]
            token_tfidf_scores = token_tfidf_scores[positive_mask]

            # Get embeddings for all valid tokens
            token_embeddings = np.array(
                [word_vectors[token] for token in valid_tokens])

            # Weight embeddings by TF-IDF scores
            total_weight = np.sum(token_tfidf_scores)
            if total_weight > 0:  # Avoid division by zero
                weighted_sum = np.sum(
                    token_embeddings * token_tfidf_scores[:, np.newaxis], axis=0)
                weighted_embeddings[doc_idx] = weighted_sum / total_weight

        return weighted_embeddings

    def calculate_centroids(self, posts_df: pd.DataFrame, embeddings_column: str) -> np.ndarray:
        """
        Calculate the centroid for each subreddit based on its embeddings.
        """
        subreddit_centroids = posts_df.groupby('subreddit')[embeddings_column].apply(
            lambda x: np.mean(np.vstack(x), axis=0))
        centroid_matrix = np.vstack(subreddit_centroids.values)

        return centroid_matrix
