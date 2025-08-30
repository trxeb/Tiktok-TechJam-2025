import pandas as pd
import numpy as np
import re
import spacy
import joblib
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Load classifier & models
# ----------------------------
clf = joblib.load("random_forest_model.pkl")  # your saved RF model
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Keywords (same as training)
# ----------------------------
PROMO_WORDS = {
    'discount', 'free', 'promo', 'coupon', 'code', 'deal', 'offer', 'save $', '% off', 
    'buy', 'purchase', 'sale', 'promotion', 'promotional', 'special offer', 'limited time', 
    'best price', 'cheap', 'cheapest', 'cheap deal', 'visit our website', 'follow us', 
    'subscribe', 'contact us', 'call now', 'sign up', 'join now', 'check out', 'shop now',
    'recommended by', 'must-have'
}

NEVER_VISITED_PHRASES = {
    "never visited", "never been", "haven't been", "haven't visited", "didn't go",
    "not been", "not go", "hasn't been", "hasn't visited", "never actually went",
    'not visited', 'someone told me', 'I read that', 'was not there', 
    'heard', 'seems', 'probably', 'think', 'might', 'looks', 
    "terrible", "worst", "awful", "scam", "cheated", "fraud","disgusting", "rude", "never again"
}

# ----------------------------
# Numeric / binary features
# ----------------------------
def link_presence(text):
    return int(bool(re.search(r"http\S+|www\S+", text)))

def uppercase_ratio(text):
    return sum(1 for c in text if c.isupper()) / max(len(text), 1)

def review_length(text):
    return len(text)

def pos_counts(text):
    doc = nlp(text)
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    adj_count = sum(1 for token in doc if token.pos_ == "ADJ")
    return noun_count, adj_count

# ----------------------------
# TF-IDF features
# ----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_features(text, keywords):
    vectorizer = TfidfVectorizer(vocabulary=keywords, ngram_range=(1,3))
    return pd.DataFrame(vectorizer.fit_transform([text]).toarray(), columns=vectorizer.get_feature_names_out())

# ----------------------------
# Embeddings
# ----------------------------
def avg_keyword_embedding(texts, keywords):
    dim = sbert_model.get_sentence_embedding_dimension()
    all_embeddings = []
    for text in texts:
        found_kw = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text.lower())]
        if found_kw:
            emb = sbert_model.encode(found_kw, show_progress_bar=False)
            avg_emb = np.mean(emb, axis=0)
        else:
            avg_emb = np.zeros(dim)
        all_embeddings.append(avg_emb)
    return np.vstack(all_embeddings)

def avg_review_embedding(texts):
    return sbert_model.encode(texts, show_progress_bar=False)

# ----------------------------
# Prepare review features
# ----------------------------
def prepare_review(review):
    df_features = pd.DataFrame()

    # Numeric / binary
    link = link_presence(review)
    upper = uppercase_ratio(review)
    length = review_length(review)
    noun_count, adj_count = pos_counts(review)
    contains_promo_kw = int(any(kw in review.lower() for kw in PROMO_WORDS))
    contains_rant_kw = int(any(kw in review.lower() for kw in NEVER_VISITED_PHRASES))

    df_features["link_presence"] = [link]
    df_features["uppercase_ratio"] = [upper]
    df_features["review_length"] = [length]
    df_features["noun_count"] = [noun_count]
    df_features["adj_count"] = [adj_count]
    df_features["contains_promo_kw"] = [contains_promo_kw]
    df_features["contains_rant_kw"] = [contains_rant_kw]

    # TF-IDF
    df_features = pd.concat([df_features, build_tfidf_features(review, PROMO_WORDS)], axis=1)
    df_features = pd.concat([df_features, build_tfidf_features(review, NEVER_VISITED_PHRASES)], axis=1)

    # Embeddings
    promo_emb = avg_keyword_embedding([review], PROMO_WORDS)
    rant_emb = avg_keyword_embedding([review], NEVER_VISITED_PHRASES)
    review_emb = avg_review_embedding([review])

    df_features = pd.concat([
        df_features,
        pd.DataFrame(promo_emb, columns=[f"promo_emb_{i}" for i in range(promo_emb.shape[1])]),
        pd.DataFrame(rant_emb, columns=[f"rant_emb_{i}" for i in range(rant_emb.shape[1])]),
        pd.DataFrame(review_emb, columns=[f"review_emb_{i}" for i in range(review_emb.shape[1])])
    ], axis=1)

    # Reindex to match training
    df_features = df_features.reindex(columns=clf.feature_names_in_, fill_value=0)
    return df_features

# ----------------------------
# Predict
# ----------------------------
def predict_review(review):
    df_feat = prepare_review(review)
    pred = clf.predict(df_feat)[0]
    return pred

# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    review_text = "I loved the service and the food was amazing!"
    label = predict_review(review_text)
    print(f"Predicted label: {label}")
