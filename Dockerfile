# Use an official lightweight Python image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first (for efficient Docker caching)
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
# Check this line: It MUST combine pip install AND cache cleanup (rm -rf)
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# Install NLTK resources right into the Docker image
# This prevents the initial download step from failing during runtime
#RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# 💥 CRITICAL FIX: Install ALL required NLTK data directly during the Docker build.
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"
ENV NLTK_DATA=/usr/local/share/nltk_data


# --- ADDED STEP: COPY THE MODEL ARTIFACT ---
# ASSUMPTION: The model files (including MLmodel) are in a subfolder named 'model_deployment_package' 
# relative to where you run docker build.
COPY model_deployment_package /app/model

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
#ENTRYPOINT ["mlflow"]

# This is what you have:
ENTRYPOINT ["mlflow"]

# You MUST add this immediately after it:
CMD ["models", "serve", "--model-uri", "file:///app/model", "--host", "0.0.0.0", "--port", "8080"]
