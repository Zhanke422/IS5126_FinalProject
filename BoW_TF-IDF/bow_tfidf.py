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

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
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
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

class FeatureEngineering:
    def __init__(self, max_features=10000):
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
        bow_features = normalize(bow_features, norm='l2')
        return bow_features.toarray()  # Convert to dense array
    
    def transform_bow(self, texts):
        """
        Transform new text using trained BoW vectorizer
        """
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        bow_features = self.bow_vectorizer.transform(processed_texts)
        bow_features = normalize(bow_features, norm='l2')
        return bow_features.toarray()  # Convert to dense array
    
    def fit_transform_tfidf(self, texts):
        """
        Preprocess text and convert to TF-IDF features
        """
        # Preprocess all texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        # Convert to TF-IDF features
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        return tfidf_features.toarray()  # Convert to dense array
    
    def transform_tfidf(self, texts):
        """
        Transform new text using trained TF-IDF vectorizer
        """
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        return tfidf_features.toarray()  # Convert to dense array
    
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
    feature_engineering = FeatureEngineering(max_features=10000)
    
    # Generate BoW features
    print("Generating BoW features...")
    bow_train = feature_engineering.fit_transform_bow(X_train)
    bow_val = feature_engineering.transform_bow(X_val)
    
    # Generate TF-IDF features
    print("Generating TF-IDF features...")
    tfidf_train = feature_engineering.fit_transform_tfidf(X_train)
    tfidf_val = feature_engineering.transform_tfidf(X_val)
    
    # Save features and labels
    print("Saving features and labels...")
    np.save('bow_features_train.npy', bow_train)
    np.save('bow_features_val.npy', bow_val)
    np.save('tfidf_features_train.npy', tfidf_train)
    np.save('tfidf_features_val.npy', tfidf_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    
    print("Feature engineering completed!") 