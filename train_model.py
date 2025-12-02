import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import os
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score

# Load environment variables from .env file
load_dotenv()

# --- 1. NLTK Setup (Path Fix) ---
# 💥 NEW CRITICAL FIX: Manually add the CI/CD download path to NLTK search paths
NLTK_DATA_PATH = '/home/runner/nltk_data'
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

# 💥 NEW STRUCTURAL FIX: Initialize global NLTK objects *after* the path has been set.
# This prevents them from failing to load their resources on import.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
    
# --- 2. Configuration ---
DATA_PATH = 'data/twitter_training.csv'
COLUMN_NAMES = ['ID', 'Entity', 'Sentiment', 'Tweet']
RANDOM_STATE = 42
MAX_ITER = 1000
# Removed redundant NLTK_DATA_PATH line here

# --- 3. Text Preprocessing Function ---
def preprocess_text(text):
    """
    Cleans and preprocesses the tweet text using the globally initialized objects.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text) # Replace non-alphabetic with space

    # 2. Tokenize, remove stopwords, and lemmatize
    tokens = [
        # Use the global lemmatizer and stop_words objects
        lemmatizer.lemmatize(word)
        for word in nltk.word_tokenize(text)
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)

# --- 4. Main Training Function ---
def train_and_log_model(data_path=DATA_PATH):
    # ... (function contents remain the same) ...

# ----------------- (Rest of the code is unchanged) -----------------
    # Set up MLflow
    mlflow.set_experiment("Ebay_Sentiment_Analysis_Project")

    # Start an MLflow run
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # --- Data Loading ---
        try:
            df = pd.read_csv(data_path, header=None, names=COLUMN_NAMES)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}. Please ensure the file is in the 'data' folder.")
            return

        print(f"Initial data shape: {df.shape}")

        # --- Data Cleaning ---
        df.dropna(subset=['Tweet', 'Sentiment'], inplace=True)
        df.drop_duplicates(inplace=True)
        df['Sentiment'] = df['Sentiment'].replace({'Neutral': 'Neutral'})

        print(f"Cleaned data shape: {df.shape}")

        # --- Data Preprocessing ---
        print("Preprocessing text data...")
        # This line now uses the globals defined after the path fix
        df['Cleaned_Tweet'] = df['Tweet'].apply(preprocess_text)

        # --- Feature/Target Split ---
        X = df['Cleaned_Tweet']
        y = df['Sentiment']

        # --- Data Split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        # Log parameters to MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("max_iter", MAX_ITER)

        # --- Model Pipeline ---
        text_clf = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
            ('clf', LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER, n_jobs=-1))
        ])

        # --- Model Training ---
        print("Starting model training...")
        text_clf.fit(X_train, y_train)
        print("Model training complete.")

        # --- Model Evaluation ---
        print("Evaluating model...")
        y_pred = text_clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_macro", f1_macro)

        # --- Model Saving/Logging ---
        mlflow.sklearn.log_model(
            sk_model=text_clf,
            artifact_path="model",
            registered_model_name="SentimentAnalysisModel"
        )
        print("Model saved to MLflow.")

if __name__ == "__main__":
    train_and_log_model(DATA_PATH)