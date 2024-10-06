#!/bin/bash

# Activate virtual environment if you're using one
# Uncomment the next line and replace with your virtual environment path if needed
# source /path/to/your/venv/bin/activate

# Upgrade pip itself
pip install --upgrade pip

# Upgrade specific packages
packages=(
    "numpy"
    "pandas"
    "plotly"
    "pytest"
    "scikit-learn"
    "scipy"
    "tqdm"
    "mkdocs"
    "mkdocs-autorefs"
    "mkdocs-awesome-pages-plugin"
    "mkdocs-get-deps"
    "mkdocs-material"
    "mkdocs-material-extensions"
    "mkdocstrings"
    "mkdocstrings-python"
)

for package in "${packages[@]}"
do
    pip install --upgrade "$package"
done

# Create a new requirements.txt with only the specified packages
for package in "${packages[@]}"
do
    pip freeze | grep -i "^$package==" >> temp_requirements.txt
done

# Sort the requirements file and remove duplicates
sort -u temp_requirements.txt > requirements.txt
rm temp_requirements.txt

echo "Specified packages have been upgraded and requirements.txt has been updated."
