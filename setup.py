from setuptools import setup, find_packages

setup(
    name="microkeras",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.5',
        'pandas==2.2.2',
        'plotly==5.15.0',
        'pytest==8.3.2',
        'scikit-learn==1.5.1',
        'scipy==1.14.1',
        'tqdm==4.66.5',
        'mkdocs==1.6.1',
        'mkdocs-autorefs==1.2.0',
        'mkdocs-awesome-pages-plugin==2.9.3',
        'mkdocs-get-deps==0.2.0',
        'mkdocs-material==9.5.39',
        'mkdocs-material-extensions==1.3.1',
        'mkdocstrings==0.26.1',
        'mkdocstrings-python==1.11.1',
    ],
)
