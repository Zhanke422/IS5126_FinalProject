# Twitter Sentiment Analysis Project

This repository contains a Twitter sentiment analysis pipeline for classifying tweets into positive, negative, or neutral sentiments.

## Project Overview

The project implements a complete NLP pipeline for sentiment analysis:

1. **Data Preprocessing**: Cleaning tweets, handling emojis, and tokenization
2. **Feature Engineering**: Traditional (BoW, TF-IDF) and modern (Word2Vec, GloVe, BERT) embeddings
3. **Model Training**: Baseline models (LR, SVM, RF) and deep learning (BiLSTM, BERT)
4. **Evaluation & Interpretability**: Performance metrics and SHAP analysis

## Project Structure

```
.
├── data/                     # Data directory
│   ├── twitter_training.csv  # Original training dataset
│   ├── twitter_validation.csv    # Original validation dataset
│   ├── features/             # Generated features
├── models/                   # Saved models and vectorizers
├── notebooks/                # Jupyter notebooks
│   ├── 01_Data_Exploration_Preprocessing.ipynb  # Data cleaning & EDA
│   ├── 02_Feature_Engineering.ipynb             # Feature generation
│   ├── 02_Word2Vec.ipynb             # Feature generation
│   ├── 03_Model_Training.ipynb                  # Model training
│   └── 04_Evaluation.ipynb     # Model evaluation
│   └── 04_Interpretability.ipynb     # Model evaluation
├── results/                  # Evaluation results and visualizations
└── README.md                 # This file
```

## Dataset

The project uses the [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) dataset from Kaggle, which contains tweets labeled with sentiment (Positive, Negative, Neutral) and an entity/category.

## Setup Instructions
1. **Environment Setup**:
   
   This project was primarily developed using Google Colab for easier reporting and resource management. The notebooks are optimized for the Colab environment but can also be run locally with some modifications.
   
   **For Google Colab**:
   - Upload the repository to Google Drive
   - Open the notebooks with Google Colab
   - The notebooks contain code for mounting Google Drive and handling paths in Colab. Please make sure that you have set up the correct path for datasets and features/models generated. 
   
   **For Local Environment**:
   - Clone the repository:
   ```
   git clone <repository-url>
   cd twitter-sentiment-analysis
   ```
   - Create a virtual environment and install dependencies:
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     pip install -r requirements.txt
     ```
   - You'll need to modify the file paths in the notebooks as they are currently set up for Colab
   - Some computationally intensive parts (like BERT fine-tuning) may require a GPU

2. **Dataset**:
   - `twitter_training.csv` and `twitter_validation.csv` is included in the repository in the `data/` directory

3. **Run the notebooks in sequence**:
   - Start with `01_Data_Exploration_Preprocessing.ipynb`
   - Continue with `02_Word2Vec.ipynb`, then `02_Feature_Engineering.ipynb`
   - Then `03_Model_Training.ipynb`
   - Finally `04_Evaluation.ipynb` and `04_Interpretability.ipynb`

## Google Colab Support

The notebooks are primarily designed for Google Colab and include specific code sections for:

1. Mounting Google Drive
2. Setting appropriate file paths for the Colab environment
3. Utilizing Colab's GPU resources for model training

When running locally, you'll need to modify these sections to work with your local file system. Comments in the notebooks indicate the sections that require modification for local execution.

## Key Features

- **Comprehensive preprocessing** for Twitter data (emojis, mentions, hashtags)
- **Multiple feature representations** (BoW, TF-IDF, Word2Vec, GloVe, BERT)
- **Baseline and advanced models** (Logistic Regression, SVM, Random Forest, BiLSTM)
- **Hyperparameter tuning** via Grid Search and Optuna
- **Model interpretability** with SHAP values
- **Detailed visualization** of results and model comparisons

## Requirements

Key libraries include:

- numpy, pandas, matplotlib, seaborn
- scikit-learn
- nltk
- gensim
- torch
- transformers (for BERT)
- shap
- optuna

## Additional Note

Except preprocessed dataset, all of the results and models are too large for submission and have been excluded from this repository. If you wish to access these files, please reach out to the project group for assistance.

## Acknowledgements

- The Twitter Entity Sentiment Analysis dataset from Kaggle
- Stanford's GloVe project for pretrained embeddings
- Hugging Face for the transformers library 