#CI Phase: Training in Github actions virtual machine
#So, while the GitHub Actions runner is the virtual machine where the code executes,
#  the setup.py file is the instruction manual that defines the command and dictates 
# how that code is executed during the automation phase.

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