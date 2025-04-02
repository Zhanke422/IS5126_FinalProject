import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
from scipy import sparse

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data if not available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
    
    def preprocess(self, text):
        """
        Preprocess the text
        """
        # Handle NaN values
        if pd.isna(text):
            return ""
            
        # Ensure text is string type
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize and remove stopwords
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

class FeatureEngineering:
    def __init__(self, max_features=10000):  # Reduce feature dimensions
        self.preprocessor = TextPreprocessor()
        self.bow_vectorizer = CountVectorizer(max_features=max_features)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    
    def fit_transform_bow(self, texts):
        """
        Preprocess text and convert to BoW features
        """
        # Preprocess all texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        # Convert to BoW features
        bow_features = self.bow_vectorizer.fit_transform(processed_texts)
        # Optional: Apply L2 normalization
        bow_features = normalize(bow_features, norm='l2', copy=False)
        return bow_features  # Keep sparse format
    
    def transform_bow(self, texts):
        """
        Transform new text using trained BoW vectorizer
        """
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        bow_features = self.bow_vectorizer.transform(processed_texts)
        bow_features = normalize(bow_features, norm='l2', copy=False)
        return bow_features  # Keep sparse format
    
    def fit_transform_tfidf(self, texts):
        """
        Preprocess text and convert to TF-IDF features
        """
        # Preprocess all texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        # Convert to TF-IDF features
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        return tfidf_features  # Keep sparse format
    
    def transform_tfidf(self, texts):
        """
        Transform new text using trained TF-IDF vectorizer
        """
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        return tfidf_features  # Keep sparse format
    
    def get_feature_names(self, feature_type='bow'):
        """
        Get feature names
        """
        if feature_type == 'bow':
            return self.bow_vectorizer.get_feature_names_out()
        else:  # tfidf
            return self.tfidf_vectorizer.get_feature_names_out()

# Usage example
if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct file paths
    train_path = os.path.join(current_dir, '..', 'twitter dataset', 'twitter_training.csv')
    val_path = os.path.join(current_dir, '..', 'twitter dataset', 'twitter_validation.csv')
    
    # Read training data
    print(f"Reading training data from: {train_path}")
    train_df = pd.read_csv(train_path, names=['id', 'category', 'sentiment', 'text'])
    
    print(f"Reading validation data from: {val_path}")
    validation_df = pd.read_csv(val_path, names=['id', 'category', 'sentiment', 'text'])
    
    # Extract text and labels
    X_train = train_df['text'].values
    y_train = train_df['sentiment'].values
    X_val = validation_df['text'].values
    y_val = validation_df['sentiment'].values
    
    # Create feature engineering object
    feature_engineering = FeatureEngineering(max_features=10000)  # Reduce feature dimensions
    
    # Generate BoW features
    print("Generating BoW features...")
    bow_train = feature_engineering.fit_transform_bow(X_train)
    bow_val = feature_engineering.transform_bow(X_val)
    
    # Generate TF-IDF features
    print("Generating TF-IDF features...")
    tfidf_train = feature_engineering.fit_transform_tfidf(X_train)
    tfidf_val = feature_engineering.transform_tfidf(X_val)
    
    # Save features and labels
    print("\nSaving features...")
    # Save complete sparse matrices
    sparse.save_npz('bow_features_train.npz', bow_train)
    sparse.save_npz('bow_features_val.npz', bow_val)
    sparse.save_npz('tfidf_features_train.npz', tfidf_train)
    sparse.save_npz('tfidf_features_val.npz', tfidf_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    
    print("\nFeature engineering completed!")
    print(f"BoW features shape - Train: {bow_train.shape}, Val: {bow_val.shape}")
    print(f"TF-IDF features shape - Train: {tfidf_train.shape}, Val: {tfidf_val.shape}")
    print(f"Labels shape - Train: {y_train.shape}, Val: {y_val.shape}")
    
    # Print sparsity information
    print("\nFeature sparsity:")
    print(f"BoW train: {1.0 - bow_train.nnz/(bow_train.shape[0]*bow_train.shape[1]):.4f}")
    print(f"BoW val: {1.0 - bow_val.nnz/(bow_val.shape[0]*bow_val.shape[1]):.4f}")
    print(f"TF-IDF train: {1.0 - tfidf_train.nnz/(tfidf_train.shape[0]*tfidf_train.shape[1]):.4f}")
    print(f"TF-IDF val: {1.0 - tfidf_val.nnz/(tfidf_val.shape[0]*tfidf_val.shape[1]):.4f}")
    
    # Print file size information
    print("\nFile sizes:")
    for file in ['bow_features_train.npz', 'bow_features_val.npz', 
                 'tfidf_features_train.npz', 'tfidf_features_val.npz']:
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"{file}: {size_mb:.2f} MB") 