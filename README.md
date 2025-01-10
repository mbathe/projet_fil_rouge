# 🍲 AAIS-AI 

### Projet BGDIA706 CREATION DE VALEUR

## Prerequisites
Before running the code, ensure the following are installed and configured:
- **Python 3.11** or higher
- **Poetry** for dependency management. [Install Poetry](https://python-poetry.org/docs/#installation)

Environment variables are stored in the `.env` file.

## Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/mbathe/projet_fil_rouge
cd projet_fil_rouge
```

### Step 2: Install Dependencies
```bash
poetry install
```

### Step 3: Download the Dataset
Run the following command at the project root to download the TIFF image dataset of chronological plant evolution and save it to the default location `./data/images/` (defined by the **DIR_DATASET** environment variable).
```bash
poetry run python scripts/download_dataset.py
```