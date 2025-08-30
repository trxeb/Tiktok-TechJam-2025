# TikTok-TechJam-2025 Review Classification

## Project Overview

This project classifies Google Local reviews into four categories: **GENUINE**, **PROMOTION**, **IRRELEVANT**, and **RANT**.  
It uses weak supervision (Snorkel labeling functions), feature engineering (TF-IDF + SVD), and a supervised Random Forest classifier.  
The pipeline enables scalable, semi-automated review moderation and analysis.

---

## Setup Instructions

1. **Clone the repository and install dependencies:**
    ```bash
    git clone <your-repo-url>
    cd TikTok-TechJam-2025
    pip install -r requirements.txt
    ```
---

## How to Reproduce Results

1. **Data Cleaning:**  
   Clean your raw review data and save as `googlelocal_reviews_cleaned.csv`.

2. **Weak Labeling:**  
   Run the label model to generate weak labels:
   ```bash
   python Data\ Labeling/label_model_training.py
   ```
   Output: `googlelocal_reviews_weakly_labeled.csv`

3. **Feature Engineering:**  
   Generate TF-IDF and SVD features:
   ```bash
   python feature_eng.py
   ```
   Output: `googlelocal_reviews_tfidf_features.csv`

4. **Model Training:**  
   Train the Random Forest classifier:
   ```bash
   python train_model.py
   ```
   Output: `review_classifier_model.pkl`

5. **Gold Set Evaluation:**  
   - Prepare `gold_set.csv` with human-labeled reviews.
   - Generate features for the gold set (using the same vectorizer/SVD as training).
   - Evaluate the model:
     ```bash
     python evaluate_gold_set.py
     ```

---

## Team Member Contributions
We are all competent and responsible team members. 
---
