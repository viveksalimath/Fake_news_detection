import DataPrep
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

print("Loading data from DataPrep module...")

# Use the original DataPrep module
train_news = DataPrep.train_news
test_news = DataPrep.test_news

print(f"Training data shape: {train_news.shape}")
print(f"Test data shape: {test_news.shape}")

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