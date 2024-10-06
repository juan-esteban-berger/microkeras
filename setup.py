from setuptools import setup, find_packages

setup(
    name="microkeras",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'mkdocs==1.6.1',
        'mkdocs-autorefs==1.2.0',
        'mkdocs-awesome-pages-plugin==2.9.3',
        'mkdocs-get-deps==0.2.0',
        'mkdocs-material==9.5.39',
        'mkdocs-material-extensions==1.3.1',
        'mkdocstrings==0.26.1',
        'mkdocstrings-python==1.11.1',
        'numpy>=1.23,<2.0',
        'pandas>=2.0,<2.3',
        'plotly==5.24.1',
        'pytest==8.3.3',
        'scikit-learn==1.5.2',
        'scipy>=1.13,<1.15',
        'tqdm==4.66.5'
    ],
)
