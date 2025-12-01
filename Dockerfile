# Use an official lightweight Python image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first (for efficient Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK resources right into the Docker image
# This prevents the initial download step from failing during runtime
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Copy the rest of the application code
# The setup.py will handle installing train_model.py and predict.py as scripts
COPY . /app

# Install the application as a package in editable mode
# This creates the 'train_sentiment' and 'predict_sentiment' scripts
RUN pip install -e .

# Define the command to run the model training script (Example: Training step in CI)
# CMD ["train_sentiment"]

# Or define the command to run the prediction service (Example: Deployment/Inference)
# CMD ["predict_sentiment"]

# Define the entry point for running MLflow locally for testing, if needed
ENTRYPOINT ["mlflow"]