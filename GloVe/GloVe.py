import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import os
import requests
import zipfile


def download_glove(glove_dir='../data', dim=50):
    os.makedirs(glove_dir, exist_ok=True)
    glove_file = os.path.join(glove_dir, f'glove.6B.{dim}d.txt')
    if not os.path.exists(glove_file):
        print("Downloading GloVe embeddings...")
        url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        local_zip = os.path.join(glove_dir, 'glove.6B.zip')
        response = requests.get(url, stream=True)
        with open(local_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)
        os.remove(local_zip)
    return glove_file


def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def tokenize_text(text):
    if not isinstance(text, str):
        return []
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words
              and len(token) > 1]
    return tokens


def tweet_to_vector(tokens, embeddings):
    vectors = []
    for token in tokens:
        if token in embeddings:
            vectors.append(embeddings[token])
    if len(vectors) == 0:
        return np.zeros(50)
    else:
        return np.mean(vectors, axis=0)


def get_glove_embedding():
    glove_file = download_glove()
    embeddings = load_glove_embeddings(glove_file)

    train_df = pd.read_csv('../twitter dataset/twitter_training.csv', nrows=100)
    val_df = pd.read_csv('../twitter dataset/twitter_validation.csv', nrows=10)

    train_df['Tokens'] = train_df['Tweet Content'].apply(tokenize_text)
    val_df['Tokens'] = val_df['Tweet Content'].apply(tokenize_text)

    train_X = np.array([tweet_to_vector(tokens, embeddings) for tokens in train_df['Tokens']])
    val_X = np.array([tweet_to_vector(tokens, embeddings) for tokens in val_df['Tokens']])
    return train_X, val_X


X1, X2 = get_glove_embedding()
print(X1[0])
print(X2[0])