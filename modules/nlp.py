import re
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NLP:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> list:
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

        return filtered_tokens


if __name__ == "__main__":
    nlp = NLP()
    text = "This is a sample text. It will be preprocessed."
    tokens = nlp.preprocess_text(text)
    print(tokens)
