from setuptools import setup, find_packages

setup(
    name='ebay_mlops_sentiment',
    version='0.0.1',
    description='MLOps project for sentiment analysis.',
    author='Gani Rao',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'mlflow',
        'nltk',
        # Add any other required packages
    ],
    # Define entry points for command-line execution
    entry_points={
        'console_scripts': [
            'train_sentiment=train_model:train_and_log_model',
            'predict_sentiment=predict:run_inference',
        ],
    },
)