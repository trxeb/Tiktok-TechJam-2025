import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# Load NLP & embedding models
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Keywords
PROMO_KEYWORDS = ["discount", "free", "promo", "offer", "deal", "sale", "voucher", "coupon", "buy one get one"]
RANT_KEYWORDS = ["never visited", "didn't go", "i heard", "looks", "planning to go", "just passing by"]

# --- Load CSV cleaned reviews --- #
def load_cleaned_csv(file_path):
    df = pd.read_csv(file_path)
    if "text_clean" in df.columns:
        return df["text_clean"].tolist()
    else:
        raise ValueError("Column 'text_clean' not found in CSV")

# --- Numeric & binary features --- #
def link_presence(text):
    return int(bool(re.search(r"http\S+|www\S+", text)))

def uppercase_ratio(text):
    if len(text) == 0:
        return 0
    upper_chars = sum(1 for c in text if c.isupper())
    return upper_chars / len(text)

def review_length(text):
    return len(text)

def pos_counts(text):
    doc = nlp(text)
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    adj_count = sum(1 for token in doc if token.pos_ == "ADJ")
    return noun_count, adj_count

# --- TF-IDF features --- #
def tfidf_features(texts, keywords):
    vectorizer = TfidfVectorizer(vocabulary=keywords)
    tfidf_matrix = vectorizer.fit_transform(texts)
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df_tfidf

# --- Embedding features --- #
def avg_keyword_embedding(texts, keywords):
    all_embeddings = []
    for text in tqdm(texts, desc="Computing keyword embeddings"):
        found_kw = [kw for kw in keywords if kw in text.lower()]
        if found_kw:
            emb = sbert_model.encode(found_kw, show_progress_bar=False)
            avg_emb = np.mean(emb, axis=0)
        else:
            avg_emb = np.zeros(sbert_model.get_sentence_embedding_dimension())
        all_embeddings.append(avg_emb)
    return np.vstack(all_embeddings)

def avg_review_embedding(texts, batch_size=64):
    # Faster batch encoding
    return sbert_model.encode(texts, batch_size=batch_size, show_progress_bar=True)

# --- Cosine similarity --- #
def cosine_similarity(a, b):
    return util.cos_sim(a, b).cpu().numpy()

# --- Build full feature set --- #
def build_features(file_path, relevant_avg_emb=None, genuine_pos_emb=None, genuine_neg_emb=None):
    texts = load_cleaned_csv(file_path)
    df = pd.DataFrame({"text": texts})
    
    # Numeric & binary features with progress bars
    df["link_presence"] = [link_presence(t) for t in tqdm(texts, desc="Link presence")]
    df["uppercase_ratio"] = [uppercase_ratio(t) for t in tqdm(texts, desc="Uppercase ratio")]
    df["review_length"] = [review_length(t) for t in tqdm(texts, desc="Review length")]
    
    pos_adj = [pos_counts(t) for t in tqdm(texts, desc="POS tagging")]
    df["noun_count"] = [x[0] for x in pos_adj]
    df["adj_count"] = [x[1] for x in pos_adj]
    
    # --- Promotional features --- #
    df_promo_tfidf = tfidf_features(texts, PROMO_KEYWORDS)
    df_promo_emb = avg_keyword_embedding(texts, PROMO_KEYWORDS)
    
    # --- Rant features --- #
    df_rant_tfidf = tfidf_features(texts, RANT_KEYWORDS)
    df_rant_emb = avg_keyword_embedding(texts, RANT_KEYWORDS)
    
    # --- Whole review embeddings --- #
    df_review_emb = avg_review_embedding(texts, batch_size=64)
    
    # --- Similarity to relevant/genuine reviews (optional) --- #
    if relevant_avg_emb is not None:
        sim_to_relevant = []
        for emb in tqdm(df_review_emb, desc="Sim to relevant"):
            sim = cosine_similarity(emb, relevant_avg_emb)
            sim_to_relevant.append(sim[0][0])
        df["sim_to_relevant"] = sim_to_relevant
    
    if genuine_pos_emb is not None and genuine_neg_emb is not None:
        sim_pos, sim_neg = [], []
        for emb in tqdm(df_review_emb, desc="Sim to genuine pos/neg"):
            sim_p = cosine_similarity(emb, genuine_pos_emb)
            sim_n = cosine_similarity(emb, genuine_neg_emb)
            sim_pos.append(sim_p[0][0])
            sim_neg.append(sim_n[0][0])
        df["sim_to_genuine_pos"] = sim_pos
        df["sim_to_genuine_neg"] = sim_neg
    
    # --- Flag likely problematic reviews --- #
    df["flag_review"] = (
        (df["link_presence"] == 1) | 
        (df["uppercase_ratio"] > 0.5) | 
        (df["review_length"] < 10) | 
        (df_promo_tfidf.sum(axis=1) > 0) | 
        (df_rant_tfidf.sum(axis=1) > 0)
    ).astype(int)
    
    # Combine all features
    features = pd.concat([
        df,
        df_promo_tfidf,
        pd.DataFrame(df_promo_emb, columns=[f"promo_emb_{i}" for i in range(df_promo_emb.shape[1])]),
        df_rant_tfidf,
        pd.DataFrame(df_rant_emb, columns=[f"rant_emb_{i}" for i in range(df_rant_emb.shape[1])]),
        pd.DataFrame(df_review_emb, columns=[f"review_emb_{i}" for i in range(df_review_emb.shape[1])])
    ], axis=1)
    
    return features

if __name__ == "__main__":
    features_df = build_features("googlelocal_reviews_cleaned.csv")
    features_df.to_csv("googlelocal_review_features_flagged.csv", index=False)
    print("Feature extraction complete! Saved to googlelocal_review_features_flagged.csv")
