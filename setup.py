from setuptools import setup, find_packages

setup(
    name='ebay_mlops_sentiment',
    version='0.0.1',
    description='MLOps project for sentiment analysis.',
    author='Ganji Rao',
    
    # 💥 FIX: Explicitly declare the root-level Python files as modules
    py_modules=['train_model', 'predict'],
    
    # packages=find_packages(), # Removed this line
    install_requires=[
        'pandas',
        'scikit-learn',
        'mlflow',
        'nltk',
    ],
    entry_points={
        'console_scripts': [
            'train_sentiment=train_model:train_and_log_model',
            'predict_sentiment=predict:run_inference',
        ],
    },
)