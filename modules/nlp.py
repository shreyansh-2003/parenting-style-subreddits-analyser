import re
import nltk
import pandas as pd
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
            raise ValueError("Input must be a string.")

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stopwords
        filtered_tokens = [
            word for word in tokens if word not in self.stop_words]

        # Combine tokens into a single string
        token_string = ' '.join(filtered_tokens)

        return token_string

    def generate_tfidf_matrix(self, texts: list, max_terms: int = 3000, min_doc_freq: int = 5):
        """
        Generate TF-IDF matrix from a list of tokenized texts.
        """
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")

        stop_words = list(set(self.stop_words))
        vectorizer = TfidfVectorizer(max_features=max_terms,
                                     min_df=min_doc_freq,
                                     stop_words=stop_words)

        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        return tfidf_matrix, feature_names


if __name__ == "__main__":
    nlp = NLP()
    text = "This is a sample text. It will be preprocessed."
    tokenized_text = nlp.tokenize_text(text)
    print(tokenized_text)
