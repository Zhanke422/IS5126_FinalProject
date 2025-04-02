# Twitter Sentiment Analysis - Feature Engineering Implementation

## Feature Engineering Methods

### 1. Bag of Words (BoW)

Text preprocessing: Clean text by removing URLs, special characters, and numbers, convert to lowercase, tokenize and lemmatize using NLTK, remove stopwords. 

Feature generation: Create vocabulary with 10,000 most frequent words, count word frequencies in each text, apply L2 normalization to feature vectors.

### 2. TF-IDF

Use same text preprocessing as BoW. Calculate Term Frequency (TF) by counting word occurrences in each document, calculate Inverse Document Frequency (IDF) using log transformation of document frequencies, combine TF and IDF values to create feature matrix.

## File Size Optimization

Original feature matrices were over 20GB due to storing complete vocabulary and using dense matrices. Optimized by:

1. Limiting vocabulary to 10,000 most frequent words
2. Using scipy.sparse for sparse matrix storage
3. Saving features in .npz format
4. Only storing non-zero elements

Final feature files reduced to hundreds of MB while maintaining feature quality.

## Performance Improvements

1. Memory:

   - Sparse matrix storage
   -  of MB
2. Processing:

   - Limited vocabulary size
   - Efficient data structures
3. Quality:

   - L2 normalization
   - Proper text cleaning
   - Lemmatization
