import mlflow
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. Configuration ---
# NOTE: This should match the name and version logged in train_model.py
MODEL_URI = "models:/SentimentAnalysisModel/1"

# --- 2. Text Preprocessing Function ---
# IMPORTANT: This must be identical to the one used in train_model.py
def preprocess_text(text):
    """Cleans and preprocesses the tweet text."""
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()
    # Ensure stopwords is downloaded (if not done manually before)
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    # 1. Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 2. Tokenize, remove stopwords, and lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in nltk.word_tokenize(text)
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)

# --- 3. Inference Function ---
def run_inference():
    """Loads the model from MLflow and makes predictions."""
    print(f"Loading model from MLflow Model Registry: {MODEL_URI}")

    try:
        # Load the model pipeline
        loaded_model = mlflow.sklearn.load_model(MODEL_URI)
    except Exception as e:
        print(f"Error loading model. Ensure the MLflow tracking server is accessible and model '{MODEL_URI}' is registered.")
        print(f"Details: {e}")
        return

    # --- Sample New Data for Prediction ---
    new_data = pd.Series([
        "This game is absolutely amazing! The graphics are stunning.",  # Positive
        "Borderlands is terrible, I hate the new update.",             # Negative
        "I feel quite neutral about the recent Nvidia product.",       # Neutral
        "The tweet is irrelevant to any sentiment."                    # Irrelevant
    ])

    print("\nOriginal Input Data:")
    print(new_data)

    # --- Preprocess ---
    print("\nPreprocessing data...")
    X_new = new_data.apply(preprocess_text)

    # --- Predict ---
    print("Making predictions...")
    predictions = loaded_model.predict(X_new)

    # --- Output ---
    results = pd.DataFrame({
        'Tweet': new_data,
        'Predicted_Sentiment': predictions
    })

    print("\n--- Prediction Results ---")
    print(results)
    
if __name__ == "__main__":
    run_inference()