import re

import nltk

# If not downloaded, uncomment the following lines to download the NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean_text(text):
    """
    Cleans Reddit text by removing URLs, @ mentions, stopwords, and non-alphabetic characters, 
    then tokenizing and rejoining the text into a cleaned string.

    Parameters:
    ----------
    text : str
        The input text from a Reddit post or comment, which may contain URLs, mentions, 
        punctuation, and other unwanted characters.

    Returns:
    -------
    str
        A cleaned and processed version of the input text, with URLs, mentions, and stopwords removed.
        The returned text is tokenized and then rejoined into a single string, containing only meaningful words.

    Steps:
    ------
    1. Convert the text to lowercase for consistency.
    2. Remove URLs and @ mentions using regular expressions.
    3. Remove non-alphabetic characters (e.g., punctuation).
    4. Tokenize the cleaned text into individual words.
    5. Filter out common stopwords that do not add meaning.
    6. Join the cleaned words back into a single string.

    Example:
    -------
    >>> clean_reddit_text("@user Check this out! https://example.com This is a sample Reddit post.")
    'check sample reddit post'
    """
    # Convert text to lowercase for uniformity
    text = text.lower()
    
    # Remove URLs using regex (patterns like http://, https://, www.)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove any remaining non-alphabetic characters (e.g., punctuation)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the cleaned text into individual words
    tokens = word_tokenize(text)
    
    # Define English stopwords and remove them from the token list
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    # Join the tokens back into a single cleaned string
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

