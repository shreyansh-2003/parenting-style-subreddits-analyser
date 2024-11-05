import re
import nltk
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)


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

        # Combine tokens into a single string
        token_string = ' '.join(filtered_tokens)

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

    def generate_tfidf_weighted_embeddings(self, tokens: list, word_vectors, feature_names: np.ndarray, tfidf_matrix):
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
            doc_vector = np.zeros(embedding_dim)
            total_weight = 0

            # Get the document's TF-IDF scores
            doc_tfidf = tfidf_matrix[doc_idx].toarray().flatten()

            # For each token in the document
            for token in tokens[doc_idx].split():
                # Get the token's TF-IDF score
                if not token in feature_names:
                    continue
                token_tfidf = doc_tfidf[feature_names.index(token)]

                # If the token is in the word vectors
                if token in word_vectors and token_tfidf > 0:
                    # Get the token's embedding
                    token_vector = word_vectors[token]

                    # Weight the embedding by the token's TF-IDF score
                    doc_vector += token_tfidf * token_vector
                    total_weight += token_tfidf

            # Normalize the document embedding by the total weight
            if total_weight > 0:
                doc_vector /= total_weight

            # Store the document embedding
            weighted_embeddings[doc_idx] = doc_vector

        return weighted_embeddings
