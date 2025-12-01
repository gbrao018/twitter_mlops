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

# --- 1. NLTK Setup (CORRECTED) ---
# We catch LookupError, which is what nltk.data.find() raises when resources are missing.

import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK resource 'stopwords' not found. Downloading...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK resource 'wordnet' not found. Downloading...")
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK resource 'punkt' not found. Downloading...")
    nltk.download('punkt')

# --- END OF NLTK SETUP ---

# --- 2. Configuration ---
# Assuming the data file is placed in the 'data' directory as per your setup
#DATA_PATH = 'data/twitter_training.csv'
DATA_PATH = 'data/twitter_training.csv'
# Column names based on the snippet provided (no header in the raw CSV)
COLUMN_NAMES = ['ID', 'Entity', 'Sentiment', 'Tweet']
# Model parameters
RANDOM_STATE = 42
MAX_ITER = 1000

# --- 3. Text Preprocessing Function ---
def preprocess_text(text):
    """
    Cleans and preprocesses the tweet text.
    - Lowercase
    - Remove special characters, numbers, and single characters
    - Tokenization and Stopword Removal
    - Lemmatization
    """
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # 1. Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text) # Replace non-alphabetic with space

    # 2. Tokenize, remove stopwords, and lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in nltk.word_tokenize(text)
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)

# --- 4. Main Training Function ---
def train_and_log_model(data_path=DATA_PATH):
    """
    Loads data, preprocesses, trains a model, and logs everything to MLflow.
    """
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
        # 1. Drop rows with missing values (especially in the 'Tweet' column)
        df.dropna(subset=['Tweet', 'Sentiment'], inplace=True)
        # 2. Drop duplicates
        df.drop_duplicates(inplace=True)
        # 3. Handle a potential "Neutral" sentiment typo (e.g., "Neutreal")
        df['Sentiment'] = df['Sentiment'].replace({'Neutral': 'Neutral'})

        print(f"Cleaned data shape: {df.shape}")

        # --- Data Preprocessing ---
        print("Preprocessing text data...")
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
        # Use a Pipeline to combine the vectorizer and the classifier
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
        # Log the complete pipeline to MLflow
        mlflow.sklearn.log_model(
            sk_model=text_clf,
            artifact_path="model",
            registered_model_name="SentimentAnalysisModel"
        )
        print("Model saved to MLflow.")

if __name__ == "__main__":
    train_and_log_model(DATA_PATH)