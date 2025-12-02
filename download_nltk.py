import nltk
import ssl
import os

# Get the path from the environment variable set in the workflow
# This variable is available because it was exported to $GITHUB_ENV in the previous step.
NLTK_DATA_PATH = os.environ.get('NLTK_DATA')

if not NLTK_DATA_PATH:
    print("Error: NLTK_DATA environment variable not set in the workflow.")
    # Exit with an error code if the path is missing
    exit(1)

# Fix for common SSL certificate verification issues on some environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # If the attribute doesn't exist, we skip the fix
    pass
else:
    # Set the default HTTPS context to unverified if the original exists
    ssl._create_default_https_context = _create_unverified_https_context

print(f"Downloading required NLTK data to: {NLTK_DATA_PATH}")

# Download the necessary resources, forcing the downloader to use the custom path
# Note: 'punkt' is often needed for tokenization, 'wordnet' for stemming/lemmatization.
nltk.download(['stopwords', 'wordnet', 'punkt'], download_dir=NLTK_DATA_PATH)

print("NLTK data download complete.")