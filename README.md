# E-commerce Review Intelligence System

A Python project for sentiment analysis and review insights on e-commerce product reviews.

## Overview

This project processes customer reviews and uses a trained machine learning model to predict sentiment and support review intelligence tasks.

## Contents

- `app.py` - Main application script for running the review intelligence system.
- `E-commerce Review Intelligence System main code.ipynb` - Jupyter notebook containing exploration, model training, and analysis steps.
- `requirements.txt` - Python package dependencies.
- `review intelligence system full main code.txt` - Backup or extended script content.
- `sentiment_model.pkl` - Serialized trained sentiment classifier.
- `tfidf_vectorizer.pkl` - Serialized TF-IDF vectorizer used for text feature extraction.
- `train.ft.txt.bz2` - Compressed training dataset file.
- `test.ft.txt.bz2` - Compressed test dataset file.

## Setup

1. Create a Python virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

- Windows PowerShell:
```powershell
venv\Scripts\Activate.ps1
```
- Windows Command Prompt:
```cmd
venv\Scripts\activate.bat
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main application with:

```bash
python app.py
```

If using the notebook, open `E-commerce Review Intelligence System main code.ipynb` in Jupyter Lab or Jupyter Notebook.

## Notes

- The project includes pre-trained model files (`sentiment_model.pkl` and `tfidf_vectorizer.pkl`) so predictions can be made without retraining.
- If you want to retrain the model, use the notebook and the provided dataset files.

## License

Add license details here if needed.
