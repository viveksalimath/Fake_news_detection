# -*- coding: utf-8 -*-
"""
Fixed version of classifier.py for current Python/sklearn versions
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Simple data loading (you'll need to adjust paths to your data files)
try:
    # Try to load data - adjust these paths as needed
    train_news = pd.read_csv('train.csv')  # Adjust path as needed
    test_news = pd.read_csv('test.csv')    # Adjust path as needed
    
    # Simple preprocessing
    train_news = train_news.dropna()
    test_news = test_news.dropna()
    
    print(f"Training data shape: {train_news.shape}")
    print(f"Test data shape: {test_news.shape}")
    print(f"Training columns: {train_news.columns.tolist()}")
    
except FileNotFoundError:
    print("Data files not found. Please ensure train.csv and test.csv are in the directory.")
    print("Looking for files in current directory...")
    import os
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print(f"CSV files found: {files}")
    exit()

# Create TF-IDF vectorizer
tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range=(1,2), use_idf=True, smooth_idf=False)

# Build logistic regression pipeline
logR_pipeline_ngram = Pipeline([
    ('LogR_tfidf', tfidf_ngram),
    ('LogR_clf', LogisticRegression(penalty="l2", C=1, max_iter=1000))
])

print("Training model...")
logR_pipeline_ngram.fit(train_news['Statement'], train_news['Label'])

print("Making predictions...")
predicted_LogR_ngram = logR_pipeline_ngram.predict(test_news['Statement'])
accuracy = np.mean(predicted_LogR_ngram == test_news['Label'])
print(f"Accuracy: {accuracy}")

# Save the model
print("Saving model...")
model_file = 'final_model_new.sav'
pickle.dump(logR_pipeline_ngram, open(model_file, 'wb'))
print(f"Model saved as {model_file}")