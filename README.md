# Tiktok-TechJam-2025

# Project Overview 
Our project implements a Multi-Layer Perceptron(MLP)-based classification pipeline to evaluate Google Location reviews by categorising them into multiple classes (GENUINE, PROMOTION, IRRELEVANT, RANT). The project includes:
1. Data Cleaning and Preprocessing
2. Feature Extraction (Text-based features + Text Embeddings)
3. MLP classifier model training
4. Evaluation of model using a gold-standard labeled dataset
5. Performance analysis of each violation, including precision, recall, F1-score, and visualisations

# Setup Instructions 
## 1. Prerequisites 
- Ensure that version of python is at least 3.10+

## 2. Clone and activate python environment 
### Clone from git if profile files are not available 
```bash
git clone https://github.com/yourusername/project-repo.git
cd project-repo
```

### Set up virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
## 3. Install all required packages 
`pip install -r requirements.txt`

## 4. Ensure that working directory contains the following:
- `mlp_model.pkl` (trained model)
- `feature_scaler.pkl` (saved feature scaler)
- `evaluation.py` (script to evaluate model)
- relevant dataset for evaluation (eg. `gold_set.csv`)


# How to reproduce results 
## 1. Open a terminal in the project directory 
## 2. Run the evaluation script:
`python evaluation.py` <br><br>
The script will:
- Load the model and feature scalar
- Extract features from dataset
- Evaluate and compute relevant metrics
- Saved evaluation results
- Create visualisations (`mlp_gold_set_confusion_matrix.png`)
