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
PROMO_KEYWORDS = [
    "buy", "purchase", "order", "sale", "discount", "deal", 
    "promo", "promotion", "promotional", "offer", "special offer",
    "coupon", "code", "promo code", "free", "giveaway", 
    "limited time", "best price", "cheap", "cheapest", "cheap deal",
    "click here", "visit our website", "follow us", "subscribe", 
    "contact us", "call now", "sign up", "join now", "check out", 
    "shop now", "recommended by", "top rated", "must-have",
    "buy one get one"
]

RANT_KEYWORDS = [
    "never visited", "didn't go", "i heard", "looks", "planning to go", 
    "just passing by", "not sure", "maybe", "probably", "heard of", 
    "thinking of visiting", "might try", "not been", "no experience", 
    "not my place", "someone told me", "doesn't seem", "would like to go", 
    "could be better", "not sure about", "i guess", "never tried"
]

# --- Load CSV cleaned reviews --- #
def load_cleaned_csv(file_path):
    df = pd.read_csv(file_path)
    if "text_clean" not in df.columns:
        raise ValueError("Column 'text_clean' not found in CSV")
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    return df

# --- Numeric & binary features --- #
def link_presence(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0
    return int(bool(re.search(r"http\S+|www\S+", text)))

def uppercase_ratio(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    upper_chars = sum(1 for c in text if c.isupper())
    return upper_chars / len(text)

def review_length(text):
    if not isinstance(text, str):
        return 0
    return len(text)

def pos_counts(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0, 0
    doc = nlp(text)
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    adj_count = sum(1 for token in doc if token.pos_ == "ADJ")
    return noun_count, adj_count

# --- TF-IDF features --- #
def tfidf_features(texts, keywords, ngram_range=(1,3)):
    vectorizer = TfidfVectorizer(vocabulary=keywords, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(texts)
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df_tfidf

# --- Embedding features --- #
def avg_keyword_embedding(texts, keywords, batch_size=32):
    all_embeddings = []
    dim = sbert_model.get_sentence_embedding_dimension()
    
    for text in tqdm(texts, desc="Computing keyword embeddings"):
        if not isinstance(text, str) or text.strip() == "":
            all_embeddings.append(np.zeros(dim))
            continue
        
        # Match keywords using word boundaries for multi-word phrases
        found_kw = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text.lower())]
        if found_kw:
            emb = sbert_model.encode(found_kw, batch_size=batch_size, show_progress_bar=False)
            avg_emb = np.mean(emb, axis=0)
        else:
            avg_emb = np.zeros(dim)
        all_embeddings.append(avg_emb)
    
    return np.vstack(all_embeddings)

def avg_review_embedding(texts, batch_size=64):
    return sbert_model.encode(texts, batch_size=batch_size, show_progress_bar=True)

# --- Cosine similarity --- #
def cosine_similarity(a, b):
    return util.cos_sim(a, b).cpu().numpy()

# --- Build full feature set --- #
def build_features(file_path, relevant_avg_emb=None, genuine_pos_emb=None, genuine_neg_emb=None):
    df_original = load_cleaned_csv(file_path)
    texts = df_original["text_clean"].tolist()
    df = pd.DataFrame({"text": texts})
    
    # Include other useful CSV columns if present
    for col in ["rating", "has_photos", "photo_count"]:
        if col in df_original.columns:
            df[col] = df_original[col]
    
    # Numeric & binary features
    df["link_presence"] = [link_presence(t) for t in tqdm(texts, desc="Link presence")]
    df["uppercase_ratio"] = [uppercase_ratio(t) for t in tqdm(texts, desc="Uppercase ratio")]
    df["review_length"] = [review_length(t) for t in tqdm(texts, desc="Review length")]
    
    pos_adj = [pos_counts(t) for t in tqdm(texts, desc="POS tagging")]
    df["noun_count"] = [x[0] for x in pos_adj]
    df["adj_count"] = [x[1] for x in pos_adj]
    
    # TF-IDF features
    df_promo_tfidf = tfidf_features(texts, PROMO_KEYWORDS, ngram_range=(1,3))
    df_rant_tfidf = tfidf_features(texts, RANT_KEYWORDS, ngram_range=(1,3))
    
    # Embedding features
    df_promo_emb = avg_keyword_embedding(texts, PROMO_KEYWORDS)
    df_rant_emb = avg_keyword_embedding(texts, RANT_KEYWORDS)
    df_review_emb = avg_review_embedding(texts, batch_size=64)
    
    # Similarity features
    if relevant_avg_emb is not None:
        df["sim_to_relevant"] = [cosine_similarity(emb, relevant_avg_emb)[0][0] for emb in tqdm(df_review_emb, desc="Sim to relevant")]
    
    if genuine_pos_emb is not None and genuine_neg_emb is not None:
        df["sim_to_genuine_pos"] = [cosine_similarity(emb, genuine_pos_emb)[0][0] for emb in tqdm(df_review_emb, desc="Sim to genuine pos")]
        df["sim_to_genuine_neg"] = [cosine_similarity(emb, genuine_neg_emb)[0][0] for emb in tqdm(df_review_emb, desc="Sim to genuine neg")]
    
    # Flag likely problematic reviews
    df["flag_review"] = (
        (df["link_presence"] == 1) |
        (df["uppercase_ratio"] > 0.5) |
        (df["review_length"] < 10) |
        (df_promo_tfidf.sum(axis=1) > 0) |
        (df_rant_tfidf.sum(axis=1) > 0)
    ).astype(int)
    
    # Combine all features into a single dataframe
    features = pd.concat([
        df,
        df_promo_tfidf,
        pd.DataFrame(df_promo_emb, columns=[f"promo_emb_{i}" for i in range(df_promo_emb.shape[1])]),
        df_rant_tfidf,
        pd.DataFrame(df_rant_emb, columns=[f"rant_emb_{i}" for i in range(df_rant_emb.shape[1])]),
        pd.DataFrame(df_review_emb, columns=[f"review_emb_{i}" for i in range(df_review_emb.shape[1])])
    ], axis=1)
    
    return features

# --- Main --- #
if __name__ == "__main__":
    features_df = build_features("googlelocal_reviews_cleaned.csv")
    features_df.to_csv("googlelocal_review_features_flagged.csv", index=False)
    print("Feature extraction complete! Saved to googlelocal_review_features_flagged.csv")
