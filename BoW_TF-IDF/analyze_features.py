import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bow_tfidf import TextPreprocessor
import os
from datetime import datetime
from scipy import sparse

def print_feature_info(features, name, output_file):
    """
    Print basic information about features
    """
    info = f"\n{name} Information:\n"
    info += f"Shape: {features.shape}\n"
    info += f"Data Type: {features.dtype}\n"
    info += f"Non-zero Element Count: {features.nnz}\n"
    info += f"Sparsity: {1.0 - features.nnz/(features.shape[0]*features.shape[1]):.4f}\n"
    info += f"Feature Value Range: [{features.data.min()}, {features.data.max()}]\n"
    output_file.write(info)
    print(info)

def analyze_features():
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(current_dir, '..', 'twitter dataset', 'twitter_training.csv')
        
        # Use fixed output file name
        output_path = os.path.join(current_dir, 'bow_tfidf_analysis.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Feature Analysis Report\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            # Read original data
            f.write("Reading training data...\n")
            train_df = pd.read_csv(train_path, names=['id', 'category', 'sentiment', 'text'])
            
            # Load generated features
            f.write("Loading features...\n")
            bow_train = sparse.load_npz('bow_features_train.npz')
            bow_val = sparse.load_npz('bow_features_val.npz')
            tfidf_train = sparse.load_npz('tfidf_features_train.npz')
            tfidf_val = sparse.load_npz('tfidf_features_val.npz')
            y_train = np.load('y_train.npy', allow_pickle=True)
            y_val = np.load('y_val.npy', allow_pickle=True)
            
            # 1. Basic Feature Information
            f.write("\n1. Basic Feature Information:\n")
            print_feature_info(bow_train, "BoW Training Features", f)
            print_feature_info(bow_val, "BoW Validation Features", f)
            print_feature_info(tfidf_train, "TF-IDF Training Features", f)
            print_feature_info(tfidf_val, "TF-IDF Validation Features", f)
            
            # 2. Label Distribution
            f.write("\n2. Label Distribution:\n")
            train_dist = np.unique(y_train, return_counts=True)
            val_dist = np.unique(y_val, return_counts=True)
            f.write(f"Training set: {dict(zip(train_dist[0], train_dist[1]))}\n")
            f.write(f"Validation set: {dict(zip(val_dist[0], val_dist[1]))}\n")
            print("Training set:", train_dist)
            print("Validation set:", val_dist)
            
            # 3. Sample Analysis
            f.write("\n3. Sample Analysis:\n")
            sample_idx = 0
            sample_info = f"Sample index: {sample_idx}\n"
            sample_info += f"Original text: {train_df['text'].iloc[sample_idx]}\n"
            sample_info += f"Sentiment label: {y_train[sample_idx]}\n"
            sample_info += f"BoW feature non-zero elements: {bow_train[sample_idx].nnz}\n"
            sample_info += f"TF-IDF feature non-zero elements: {tfidf_train[sample_idx].nnz}\n"
            f.write(sample_info)
            print(sample_info)
            
            # 4. Feature Sparsity Analysis
            f.write("\n4. Feature Sparsity Analysis:\n")
            bow_sparsity = 1.0 - bow_train.nnz/(bow_train.shape[0]*bow_train.shape[1])
            tfidf_sparsity = 1.0 - tfidf_train.nnz/(tfidf_train.shape[0]*tfidf_train.shape[1])
            sparsity_info = f"BoW feature sparsity: {bow_sparsity:.4f}\n"
            sparsity_info += f"TF-IDF feature sparsity: {tfidf_sparsity:.4f}\n"
            f.write(sparsity_info)
            print(sparsity_info)
            
            # 5. Feature Consistency Check
            f.write("\n5. Feature Consistency Check:\n")
            consistency_info = f"BoW training and validation feature dimensions match: {bow_train.shape[1] == bow_val.shape[1]}\n"
            consistency_info += f"TF-IDF training and validation feature dimensions match: {tfidf_train.shape[1] == tfidf_val.shape[1]}\n"
            f.write(consistency_info)
            print(consistency_info)
            
            # 6. Feature Effectiveness Analysis
            f.write("\n6. Feature Effectiveness Analysis:\n")
            bow_zero_cols = np.array(bow_train.sum(axis=0)).flatten() == 0
            tfidf_zero_cols = np.array(tfidf_train.sum(axis=0)).flatten() == 0
            effectiveness_info = f"BoW zero feature count: {np.sum(bow_zero_cols)}\n"
            effectiveness_info += f"TF-IDF zero feature count: {np.sum(tfidf_zero_cols)}\n"
            f.write(effectiveness_info)
            print(effectiveness_info)
            
            # 7. Feature Distribution Analysis
            f.write("\n7. Feature Distribution Analysis:\n")
            bow_mean = np.array(bow_train.mean(axis=0)).flatten()
            tfidf_mean = np.array(tfidf_train.mean(axis=0)).flatten()
            
            f.write("BoW feature statistics:\n")
            bow_stats = f"  Mean: {np.mean(bow_mean):.4f}\n"
            bow_stats += f"  Standard deviation: {np.std(bow_mean):.4f}\n"
            bow_stats += f"  Median: {np.median(bow_mean):.4f}\n"
            f.write(bow_stats)
            print(bow_stats)
            
            f.write("\nTF-IDF feature statistics:\n")
            tfidf_stats = f"  Mean: {np.mean(tfidf_mean):.4f}\n"
            tfidf_stats += f"  Standard deviation: {np.std(tfidf_mean):.4f}\n"
            tfidf_stats += f"  Median: {np.median(tfidf_mean):.4f}\n"
            f.write(tfidf_stats)
            print(tfidf_stats)
            
            # 8. File Size Information
            f.write("\n8. File Size Information:\n")
            for file in ['bow_features_train.npz', 'bow_features_val.npz', 
                        'tfidf_features_train.npz', 'tfidf_features_val.npz']:
                size_mb = os.path.getsize(file) / (1024 * 1024)
                file_info = f"{file}: {size_mb:.2f} MB\n"
                f.write(file_info)
                print(file_info)
            
            f.write("\nAnalysis completed successfully!")
            print(f"\nAnalysis results have been saved to: {output_path}")
            
    except Exception as e:
        error_msg = f"Error occurred: {str(e)}\n"
        error_msg += "\nPlease make sure you have run bow_tfidf.py first to generate the feature files.\n"
        error_msg += "The required files are:\n"
        error_msg += "- bow_features_train.npz\n"
        error_msg += "- bow_features_val.npz\n"
        error_msg += "- tfidf_features_train.npz\n"
        error_msg += "- tfidf_features_val.npz\n"
        error_msg += "- y_train.npy\n"
        error_msg += "- y_val.npy\n"
        print(error_msg)
        if 'f' in locals() and not f.closed:
            f.write(error_msg)

if __name__ == "__main__":
    analyze_features()