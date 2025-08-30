import joblib
import pandas as pd

# Load vectorizer, SVD, and model
tfidf = joblib.load("tfidf_vectorizer.pkl")
svd = joblib.load("svd_transformer.pkl")
model = joblib.load("review_classifier_model.pkl")

# Label mapping
label_names = {
    0: "GENUINE", 
    1: "PROMOTION",
    2: "IRRELEVANT", 
    3: "RANT"
}

def predict_review(text):
    text_clean = str(text).strip()
    X_tfidf = tfidf.transform([text_clean])
    X_svd = svd.transform(X_tfidf)
    # Add extra features as needed
    text_length = len(text_clean)
    # Combine SVD features and extra features
    import numpy as np
    X_full = np.hstack([X_svd, [[text_length]]])  # shape (1, 101)
    pred = model.predict(X_full)[0]
    print(f"Input: {text}")
    print(f"Predicted label: {label_names.get(pred, pred)}")

if __name__ == "__main__":
    # Example usage
    sample = "Promotion at cheap deals"
    predict_review(sample)