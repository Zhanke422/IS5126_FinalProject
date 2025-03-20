import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# 1. Data Reading and Cleaning
print("Reading dataset...")
df = pd.read_csv('C:/Users/Lenovo/Dropbox/Documents/Class Group Homework/IS5126/Final_Project/twitter_training.csv', header=None)
df.columns = ['ID', 'Entity', 'Sentiment', 'Content']
print(f"Dataset shape: {df.shape}")

print("Cleaning data...")
# Delete rows with empty Content, and ensure Content is string
df = df.dropna(subset=['Content'])
df['Content'] = df['Content'].astype(str)

# 2. Preprocessing Improvements: Negation Handling and Sentiment Word Enhancement
def handle_negations(text):
    """
    Process negation words, e.g. "not good" becomes "NEG_good"
    """
    patterns = [r"\bnot\s+(\w+)", r"n't\s+(\w+)"]
    for pattern in patterns:
        text = re.sub(pattern, r"NEG_\1", text)
    return text

def preprocess_text(text):
    """
    Text preprocessing:
    1. Convert to lowercase
    2. Handle negations
    3. Remove URLs, HTML tags, special characters
    4. Tokenize
    5. Remove stopwords
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = handle_negations(text)

    # Remove URLs and HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Keep letters and some special punctuation
    text = re.sub(r'[^a-zA-Z\s:;!)(]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    return tokens

print("Preprocessing text data...")
df['Tokens'] = df['Content'].apply(preprocess_text)

# Delete rows with empty token lists
df = df[df['Tokens'].map(len) > 0]
print(f"Dataset shape after preprocessing: {df.shape}")

print("Building bigram features...")
phrases = Phrases(df['Tokens'], min_count=1, threshold=2)  
bigram = Phraser(phrases)
df['Tokens'] = df['Tokens'].apply(lambda tokens: bigram[tokens])

print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    vector_size=200,  # Word vector dimension
    window=8,         
    min_count=2,      
    negative=15,      
    hs=1,             
    sg=1,             
    alpha=0.025,      
    min_alpha=0.0001, 
    workers=4
)

# Train model
tokens_list = df['Tokens'].tolist()
word2vec_model.build_vocab(tokens_list)
word2vec_model.train(tokens_list, total_examples=word2vec_model.corpus_count, epochs=25)
word2vec_model.save("word2vec_twitter.model")
print("Word2Vec model has been saved as 'word2vec_twitter.model'")

# 5. Generate Document Vectors (Mean Word Vectors)
def get_avg_word2vec(tokens, model):
    """
    Calculate the average word vector for a document
    """
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

print("Calculating document vectors...")
X_features = np.array([get_avg_word2vec(tokens, word2vec_model) for tokens in df['Tokens']])
print("Feature engineering completed!")

np.save("word2vec_features.npy", X_features)
np.save("sentiment_labels.npy", np.array(df['Sentiment']))
print("Feature vectors and labels have been saved to 'word2vec_features.npy' and 'sentiment_labels.npy'")
