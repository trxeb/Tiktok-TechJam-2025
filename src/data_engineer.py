import pandas as pd
import sys 
from pathlib import Path
import numpy as np
import re
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------
# Load NLP & embedding models
# ----------------------------
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Import labeling-function resources from your labeling_functions.py
# ----------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from labelling.labelling_functions import (
        PROMO_WORDS, NEVER_VISITED_PHRASES, STRONG_RANT_WORDS, 
        DOMAIN_TERMS, WHITELIST_COMMON, category_to_family,
        FAMILY_MAP, ABSTAIN, GENUINE, PROMOTION, IRRELEVANT, RANT,
        # Import the regex patterns from your LFs
        NEVER_EXPLICIT_RX, HEARSAY_RX, DRIVEBY_RX, PHONE_ONLY_RX, 
        CLOSED_NO_ENTRY_RX, POLITICS_RX, EMPLOYMENT_RX, GENERIC_TEMPLATE_RX
    )
except ImportError:
    # Fallback definitions if import fails
    PROMO_WORDS = {
        'discount', 'free', 'promo', 'coupon', 'code', 'deal', 'offer', 'save'}

# Additional keyword sets for comprehensive feature engineering
GENUINE_KEYWORDS = {
    'delicious', 'tasty', 'fresh', 'crispy', 'tender', 'juicy', 'friendly', 'helpful',
    'clean', 'comfortable', 'spacious', 'cozy', 'atmosphere', 'ambiance', 'service',
    'staff', 'waitress', 'waiter', 'server', 'ordered', 'ate', 'visited', 'went'
}

IRRELEVANT_KEYWORDS = {
    'election', 'president', 'congress', 'politics', 'democrat', 'republican',
    'hiring', 'job', 'resume', 'salary', 'career', 'vs', 'compared to', 'better than'
}

# Define SPAM_KEYWORDS and FIRST_PERSON_VISIT_VERBS since they don't exist in your labeling functions
SPAM_KEYWORDS = {
    'click here', 'visit website', 'call now', 'limited time', 'act fast',
    'don\'t miss', 'hurry', 'instant', 'guarantee', 'risk free'
}

FIRST_PERSON_VISIT_VERBS = {
    'i went', 'i visited', 'i ate', 'i ordered', 'i had', 'we went', 
    'we visited', 'we ate', 'we ordered', 'we had', 'my experience', 'our visit'
}

# Regex patterns (define if not imported)
try:
    NEVER_EXPLICIT_RX
except NameError:
    NEVER_EXPLICIT_RX = re.compile(
        r"(never\s+(been|visited)|haven'?t\s+(been|visited)|didn'?t\s+(go|visit)|"
        r"\bnot\s+(been|go|visit)\b|was(n't| not)\s+there|"
        r"\bnever\s+actually\s+(went|visited)\b)",
        re.I
    )

try:
    HEARSAY_RX
except NameError:
    HEARSAY_RX = re.compile(
        r"\b(i\s+heard|someone\s+said|people\s+say|they\s+say|"
        r"i\s+read\s+(online|about)|rumor(ed)?)\b", re.I
    )

# Define other regex patterns if they don't exist
try:
    DRIVEBY_RX
except NameError:
    DRIVEBY_RX = re.compile(r"\b(drove\s+by|drove\s+past|passed\s+by|just\s+saw\s+it)\b", re.I)

try:
    PHONE_ONLY_RX
except NameError:
    PHONE_ONLY_RX = re.compile(
        r"\b(called|phoned|on\s+the\s+phone|left\s+a\s+message|voicemail|"
        r"no\s+answer|line\s+was\s+busy)\b", re.I
    )

try:
    CLOSED_NO_ENTRY_RX
except NameError:
    CLOSED_NO_ENTRY_RX = re.compile(
        r"\b(closed\s+when\s+we\s+arrived|could(n't| not)\s+get\s+in|"
        r"no\s+reservation|turn(ed)?\s+away)\b", re.I
    )

try:
    POLITICS_RX
except NameError:
    POLITICS_RX = re.compile(r"\b(election|president|congress|parliament|democrat|republican|left|right|policy|politics)\b", re.I)

try:
    EMPLOYMENT_RX
except NameError:
    EMPLOYMENT_RX = re.compile(r"\b(hiring|apply|resume|vacancy|job\s+opening|career|recruit|salary|benefit)\b", re.I)

try:
    GENERIC_TEMPLATE_RX
except NameError:
    GENERIC_TEMPLATE_RX = re.compile(r"^(nice|great|good|amazing)\s+(place|food|service)$", re.I)

# Define missing functions if they don't exist
try:
    category_to_family
except NameError:
    def category_to_family(cat_raw: str) -> str:
        """Fallback category mapping function"""
        if not cat_raw:
            return "generic"
        return "generic"

try:
    DOMAIN_TERMS
except NameError:
    DOMAIN_TERMS = {"generic": set()}

try:
    WHITELIST_COMMON
except NameError:
    WHITELIST_COMMON = {
        "good","great","nice","amazing","awesome","love","like","service","place",
        "staff","people","time","today","yesterday","experience","best","better",
        "bad","okay","ok","friendly","helpful","clean","dirty","fast","slow"
    }

# ----------------------------
# CSV loader
# ----------------------------
def load_cleaned_csv(file_path):
    df = pd.read_csv(file_path)
    if "text_clean" not in df.columns:
        raise ValueError("Column 'text_clean' not found in CSV")
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    return df

# ----------------------------
# Enhanced feature extraction functions
# ----------------------------
def extract_basic_features(texts):
    """Extract basic numerical and binary features"""
    features = {}
    
    # Link and contact information
    features['has_url'] = [int(bool(re.search(r"http\S+|www\S+", str(text)))) for text in texts]
    features['has_email'] = [int(bool(re.search(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', str(text)))) for text in texts]
    features['has_phone'] = [int(bool(re.search(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}', str(text)))) for text in texts]
    
    # Text characteristics
    features['char_count'] = [len(str(text)) for text in texts]
    features['word_count'] = [len(re.findall(r'\b\w+\b', str(text))) for text in texts]
    features['sentence_count'] = [len(re.split(r'[.!?]+', str(text))) for text in texts]
    features['avg_word_length'] = [np.mean([len(word) for word in re.findall(r'\b\w+\b', str(text))]) if re.findall(r'\b\w+\b', str(text)) else 0 for text in texts]
    features['uppercase_ratio'] = [sum(1 for c in str(text) if c.isupper()) / len(str(text)) if str(text) else 0.0 for text in texts]
    features['punctuation_ratio'] = [sum(1 for c in str(text) if c in '!@#$%^&*()') / len(str(text)) if str(text) else 0.0 for text in texts]
    
    return features

def extract_pos_features(texts):
    """Extract part-of-speech features"""
    features = {'noun_count': [], 'adj_count': [], 'verb_count': [], 'adv_count': []}
    
    for text in tqdm(texts, desc="POS tagging"):
        if not str(text).strip():
            features['noun_count'].append(0)
            features['adj_count'].append(0)
            features['verb_count'].append(0)
            features['adv_count'].append(0)
            continue
            
        doc = nlp(str(text))
        features['noun_count'].append(sum(1 for token in doc if token.pos_ == "NOUN"))
        features['adj_count'].append(sum(1 for token in doc if token.pos_ == "ADJ"))
        features['verb_count'].append(sum(1 for token in doc if token.pos_ == "VERB"))
        features['adv_count'].append(sum(1 for token in doc if token.pos_ == "ADV"))
    
    return features

def extract_keyword_features(texts):
    """Extract keyword-based features for all label categories"""
    features = {}
    
    # PROMOTION features
    features['promo_word_count'] = [sum(1 for word in PROMO_WORDS if word.lower() in str(text).lower()) for text in texts]
    features['has_social_media'] = [int(bool(re.search(r"@\w+|\binstagram|facebook|tiktok|follow\s+us\b", str(text).lower()))) for text in texts]
    
    # GENUINE features
    features['genuine_word_count'] = [sum(1 for word in GENUINE_KEYWORDS if word.lower() in str(text).lower()) for text in texts]
    features['has_first_person'] = [int(bool(re.search(r"\b(i|we|my|our)\b", str(text).lower()))) for text in texts]
    features['has_past_tense'] = [int(bool(re.search(r"\b(was|were|went|visited|ate|had|ordered)\b", str(text).lower()))) for text in texts]
    features['first_person_past_combo'] = [int(fp and pt) for fp, pt in zip(features['has_first_person'], features['has_past_tense'])]
    
    # RANT features  
    features['rant_word_count'] = [sum(1 for word in NEVER_VISITED_PHRASES if word.lower() in str(text).lower()) for text in texts]
    features['strong_rant_count'] = [sum(1 for word in STRONG_RANT_WORDS if word.lower() in str(text).lower()) for text in texts]
    features['never_visited_explicit'] = [int(bool(NEVER_EXPLICIT_RX.search(str(text)))) for text in texts]
    features['hearsay_pattern'] = [int(bool(HEARSAY_RX.search(str(text)))) for text in texts]
    
    # IRRELEVANT features
    features['irrelevant_word_count'] = [sum(1 for word in IRRELEVANT_KEYWORDS if word.lower() in str(text).lower()) for text in texts]
    features['has_politics'] = [int(bool(re.search(r"\b(election|president|congress|parliament|democrat|republican|politics)\b", str(text).lower()))) for text in texts]
    features['has_employment'] = [int(bool(re.search(r"\b(hiring|apply|resume|vacancy|job\s+opening|career|recruit|salary)\b", str(text).lower()))) for text in texts]
    features['generic_template'] = [int(bool(re.match(r"^(nice|great|good|amazing)\s+(place|food|service)$", str(text).lower().strip()))) for text in texts]
    
    return features

def extract_domain_features(df):
    """Extract domain-specific features based on business category"""
    features = {'domain_hit_ratio': [], 'off_domain_ratio': []}
    
    for _, row in df.iterrows():
        text = str(row.get('text_clean', '')).lower()
        cat = str(row.get('category_primary', '') or row.get('category_all', ''))
        
        try:
            fam = category_to_family(cat)
        except:
            fam = "generic"
            
        if fam == "generic" or fam not in DOMAIN_TERMS:
            features['domain_hit_ratio'].append(0.0)
            features['off_domain_ratio'].append(0.0)
            continue
            
        toks = re.findall(r"\b[a-z]+\b", text)
        if not toks:
            features['domain_hit_ratio'].append(0.0)
            features['off_domain_ratio'].append(0.0)
            continue
            
        try:
            dom = DOMAIN_TERMS.get(fam, set())
            toks_filtered = [t for t in toks if t not in WHITELIST_COMMON]
            
            if not toks_filtered:
                features['domain_hit_ratio'].append(0.0)
                features['off_domain_ratio'].append(0.0)
                continue
                
            domain_hits = sum(1 for t in toks_filtered if t in dom)
            off_hits = sum(1 for t in toks_filtered if t not in dom)
            
            features['domain_hit_ratio'].append(domain_hits / len(toks_filtered))
            features['off_domain_ratio'].append(off_hits / len(toks_filtered))
        except:
            features['domain_hit_ratio'].append(0.0)
            features['off_domain_ratio'].append(0.0)
    
    return features

def extract_tfidf_features(texts, keyword_sets, max_features=50):
    """Extract TF-IDF features for different keyword categories"""
    tfidf_features = {}
    
    for category, keywords in keyword_sets.items():
        if not keywords:
            continue
            
        vectorizer = TfidfVectorizer(
            vocabulary=list(keywords),
            ngram_range=(1, 2),
            max_features=max_features
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([str(text) for text in texts])
            feature_names = [f"{category}_tfidf_{name}" for name in vectorizer.get_feature_names_out()]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
            tfidf_features.update(tfidf_df.to_dict('list'))
        except:
            # Handle case where no keywords are found
            for kw in list(keywords)[:max_features]:
                tfidf_features[f"{category}_tfidf_{kw}"] = [0.0] * len(texts)
    
    return tfidf_features

def extract_embedding_features(texts, batch_size=64):
    """Extract sentence embedding features"""
    embeddings = sbert_model.encode([str(text) for text in texts], 
                                   batch_size=batch_size, 
                                   show_progress_bar=True)
    
    # Create feature names
    feature_names = [f"embedding_{i}" for i in range(embeddings.shape[1])]
    
    return {name: embeddings[:, i].tolist() for i, name in enumerate(feature_names)}

def extract_metadata_features(df):
    """Extract features from review metadata"""
    features = {}
    
    # Rating-based features
    if 'rating' in df.columns:
        features['rating'] = df['rating'].fillna(0).tolist()
        features['high_rating'] = (df['rating'] >= 4).astype(int).tolist()
        features['low_rating'] = (df['rating'] <= 2).astype(int).tolist()
    
    # Photo-based features
    if 'has_photos' in df.columns:
        features['has_photos'] = df['has_photos'].fillna(0).astype(int).tolist()
    if 'photo_count' in df.columns:
        features['photo_count'] = df['photo_count'].fillna(0).tolist()
        features['multiple_photos'] = (df['photo_count'] > 1).astype(int).tolist()
    
    # Response features
    if 'has_response' in df.columns:
        features['has_response'] = df['has_response'].fillna(0).astype(int).tolist()
    
    return features

def create_interaction_features(feature_dict):
    """Create interaction features between different categories"""
    interactions = {}
    
    # Interaction between rating and text length
    if 'rating' in feature_dict and 'word_count' in feature_dict:
        interactions['rating_length_interaction'] = [
            r * wc for r, wc in zip(feature_dict['rating'], feature_dict['word_count'])
        ]
    
    # Interaction between first person and domain relevance
    if 'has_first_person' in feature_dict and 'domain_hit_ratio' in feature_dict:
        interactions['first_person_domain_interaction'] = [
            fp * dhr for fp, dhr in zip(feature_dict['has_first_person'], feature_dict['domain_hit_ratio'])
        ]
    
    # Interaction between photos and word count
    if 'has_photos' in feature_dict and 'word_count' in feature_dict:
        interactions['photos_length_interaction'] = [
            hp * wc for hp, wc in zip(feature_dict['has_photos'], feature_dict['word_count'])
        ]
    
    return interactions

def build_comprehensive_features(file_path, include_embeddings=True):
    """Build comprehensive feature set for all label categories"""
    print("Loading data...")
    df = load_cleaned_csv(file_path)
    texts = df["text_clean"].tolist()
    
    # Initialize feature dictionary
    all_features = {'text_clean': texts}
    
    # Copy relevant original columns
    for col in ['rating', 'has_photos', 'photo_count', 'has_response', 'category_primary', 'category_all']:
        if col in df.columns:
            all_features[col] = df[col].tolist()
    
    # Extract basic features
    print("Extracting basic features...")
    basic_features = extract_basic_features(texts)
    all_features.update(basic_features)
    
    # Extract POS features
    print("Extracting POS features...")
    pos_features = extract_pos_features(texts)
    all_features.update(pos_features)
    
    # Extract keyword features
    print("Extracting keyword features...")
    keyword_features = extract_keyword_features(texts)
    all_features.update(keyword_features)
    
    # Extract domain features
    print("Extracting domain features...")
    domain_features = extract_domain_features(df)
    all_features.update(domain_features)
    
    # Extract metadata features
    print("Extracting metadata features...")
    metadata_features = extract_metadata_features(df)
    all_features.update(metadata_features)
    
    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    keyword_sets = {
        'promotion': PROMO_WORDS,
        'genuine': GENUINE_KEYWORDS,
        'rant': NEVER_VISITED_PHRASES | STRONG_RANT_WORDS,
        'irrelevant': IRRELEVANT_KEYWORDS
    }
    tfidf_features = extract_tfidf_features(texts, keyword_sets)
    all_features.update(tfidf_features)
    
    # Extract embedding features (optional due to computational cost)
    if include_embeddings:
        print("Extracting embedding features...")
        embedding_features = extract_embedding_features(texts)
        all_features.update(embedding_features)
    
    # Create interaction features
    print("Creating interaction features...")
    interaction_features = create_interaction_features(all_features)
    all_features.update(interaction_features)
    
    # Create composite flags for each label category
    print("Creating composite label flags...")
    n_samples = len(texts)
    
    # PROMOTION flags
    all_features['promotion_flag'] = [
        int(any([
            all_features['has_url'][i],
            all_features['has_email'][i], 
            all_features['has_phone'][i],
            all_features['promo_word_count'][i] > 0,
            all_features['has_social_media'][i]
        ])) for i in range(n_samples)
    ]
    
    # GENUINE flags
    all_features['genuine_flag'] = [
        int(any([
            all_features['first_person_past_combo'][i],
            all_features['genuine_word_count'][i] >= 2,
            all_features['domain_hit_ratio'][i] > 0.2,
            (all_features.get('high_rating', [0]*n_samples)[i] and all_features['word_count'][i] >= 10)
        ])) for i in range(n_samples)
    ]
    
    # RANT flags
    all_features['rant_flag'] = [
        int(any([
            all_features['never_visited_explicit'][i],
            all_features['hearsay_pattern'][i],
            all_features['strong_rant_count'][i] > 0,
            all_features['rant_word_count'][i] > 0
        ])) for i in range(n_samples)
    ]
    
    # IRRELEVANT flags
    all_features['irrelevant_flag'] = [
        int(any([
            all_features['has_politics'][i],
            all_features['has_employment'][i],
            all_features['generic_template'][i],
            all_features['off_domain_ratio'][i] > 0.8,
            all_features['word_count'][i] < 3
        ])) for i in range(n_samples)
    ]
    
    # Overall quality score
    all_features['quality_score'] = [
        all_features['genuine_flag'][i] * 2 + 
        all_features['domain_hit_ratio'][i] * 3 +
        (1 if all_features['word_count'][i] >= 10 else 0) +
        all_features.get('has_photos', [0]*n_samples)[i] -
        all_features['promotion_flag'][i] * 2 -
        all_features['rant_flag'][i] -
        all_features['irrelevant_flag'][i]
        for i in range(n_samples)
    ]
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    print(f"Feature extraction complete! Generated {len(features_df.columns)} features for {len(features_df)} samples.")
    return features_df

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    # Build comprehensive features
    features_df = build_comprehensive_features(
        "googlelocal_reviews_cleaned.csv", 
        include_embeddings=True  # Set to False for faster processing
    )
    
    # Save results
    output_file = "googlelocal_review_features_comprehensive.csv"
    features_df.to_csv(output_file, index=False)
    
    print(f"\nFeature extraction complete!")
    print(f"Saved {len(features_df)} samples with {len(features_df.columns)} features to {output_file}")
    print("\nFeature summary:")
    print(f"- Basic features: character/word counts, ratios")
    print(f"- POS features: noun, adjective, verb, adverb counts")
    print(f"- Keyword features: category-specific word counts")
    print(f"- Domain features: business category relevance")
    print(f"- TF-IDF features: weighted term frequencies")
    print(f"- Embedding features: sentence-level representations")
    print(f"- Metadata features: ratings, photos, responses")
    print(f"- Interaction features: cross-feature combinations")
    print(f"- Label flags: composite indicators for each category")
    
    # Display feature distribution
    print(f"\nLabel flag distributions:")
    for flag in ['promotion_flag', 'genuine_flag', 'rant_flag', 'irrelevant_flag']:
        if flag in features_df.columns:
            count = features_df[flag].sum()
            pct = (count / len(features_df)) * 100
            print(f"- {flag}: {count} samples ({pct:.1f}%)")
            '% off',
        'buy', 'purchase', 'sale', 'promotion', 'promotional', 'special offer', 'limited time',
        'best price', 'cheap', 'cheapest', 'cheap deal', 'visit our website', 'follow us',
        'subscribe', 'contact us', 'call now', 'sign up', 'join now', 'check out', 'shop now',
        'recommended by', 'must-have'
    
    
    NEVER_VISITED_PHRASES = {
        "never visited", "never been", "haven't been", "haven't visited", "didn't go",
        "not been", "not go", "hasn't been", "hasn't visited", "never actually went",
        "not visited", "someone told me", "i read that", "was not there", "i heard",
        "seems", "probably", "think", "might", "looks"
    }
    
    STRONG_RANT_WORDS = {
        "terrible", "worst", "awful", "scam", "cheated", "fraud", "disgusting", "rude", "never again"
    }
    
    # Define missing constants
    ABSTAIN = -1
    GENUINE = 0
    PROMOTION = 1
    IRRELEVANT = 2
    RANT = 3

# Additional keyword sets for comprehensive feature engineering
GENUINE_KEYWORDS = {
    'delicious', 'tasty', 'fresh', 'crispy', 'tender', 'juicy', 'friendly', 'helpful',
    'clean', 'comfortable', 'spacious', 'cozy', 'atmosphere', 'ambiance', 'service',
    'staff', 'waitress', 'waiter', 'server', 'ordered', 'ate', 'visited', 'went'
}

IRRELEVANT_KEYWORDS = {
    'election', 'president', 'congress', 'politics', 'democrat', 'republican',
    'hiring', 'job', 'resume', 'salary', 'career', 'vs', 'compared to', 'better than'
}

SPAM_KEYWORDS = {
    'click here', 'visit website', 'call now', 'limited time', 'act fast',
    'don\'t miss', 'hurry', 'instant', 'guarantee', 'risk free'
}

# Regex patterns (define if not imported)
try:
    NEVER_EXPLICIT_RX
except NameError:
    NEVER_EXPLICIT_RX = re.compile(
        r"(never\s+(been|visited)|haven'?t\s+(been|visited)|didn'?t\s+(go|visit)|"
        r"\bnot\s+(been|go|visit)\b|was(n't| not)\s+there|"
        r"\bnever\s+actually\s+(went|visited)\b)",
        re.I
    )

try:
    HEARSAY_RX
except NameError:
    HEARSAY_RX = re.compile(
        r"\b(i\s+heard|someone\s+said|people\s+say|they\s+say|"
        r"i\s+read\s+(online|about)|rumor(ed)?)\b", re.I
    )

# ----------------------------
# CSV loader
# ----------------------------
def load_cleaned_csv(file_path):
    df = pd.read_csv(file_path)
    if "text_clean" not in df.columns:
        raise ValueError("Column 'text_clean' not found in CSV")
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    return df

# ----------------------------
# Enhanced feature extraction functions
# ----------------------------
def extract_basic_features(texts):
    """Extract basic numerical and binary features"""
    features = {}
    
    # Link and contact information
    features['has_url'] = [int(bool(re.search(r"http\S+|www\S+", str(text)))) for text in texts]
    features['has_email'] = [int(bool(re.search(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', str(text)))) for text in texts]
    features['has_phone'] = [int(bool(re.search(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}', str(text)))) for text in texts]
    
    # Text characteristics
    features['char_count'] = [len(str(text)) for text in texts]
    features['word_count'] = [len(re.findall(r'\b\w+\b', str(text))) for text in texts]
    features['sentence_count'] = [len(re.split(r'[.!?]+', str(text))) for text in texts]
    features['avg_word_length'] = [np.mean([len(word) for word in re.findall(r'\b\w+\b', str(text))]) if re.findall(r'\b\w+\b', str(text)) else 0 for text in texts]
    features['uppercase_ratio'] = [sum(1 for c in str(text) if c.isupper()) / len(str(text)) if str(text) else 0.0 for text in texts]
    features['punctuation_ratio'] = [sum(1 for c in str(text) if c in '!@#$%^&*()') / len(str(text)) if str(text) else 0.0 for text in texts]
    
    return features

def extract_pos_features(texts):
    """Extract part-of-speech features"""
    features = {'noun_count': [], 'adj_count': [], 'verb_count': [], 'adv_count': []}
    
    for text in tqdm(texts, desc="POS tagging"):
        if not str(text).strip():
            features['noun_count'].append(0)
            features['adj_count'].append(0)
            features['verb_count'].append(0)
            features['adv_count'].append(0)
            continue
            
        doc = nlp(str(text))
        features['noun_count'].append(sum(1 for token in doc if token.pos_ == "NOUN"))
        features['adj_count'].append(sum(1 for token in doc if token.pos_ == "ADJ"))
        features['verb_count'].append(sum(1 for token in doc if token.pos_ == "VERB"))
        features['adv_count'].append(sum(1 for token in doc if token.pos_ == "ADV"))
    
    return features

def extract_keyword_features(texts):
    """Extract keyword-based features for all label categories"""
    features = {}
    
    # PROMOTION features
    features['promo_word_count'] = [sum(1 for word in PROMO_WORDS if word.lower() in str(text).lower()) for text in texts]
    features['has_social_media'] = [int(bool(re.search(r"@\w+|\binstagram|facebook|tiktok|follow\s+us\b", str(text).lower()))) for text in texts]
    
    # GENUINE features
    features['genuine_word_count'] = [sum(1 for word in GENUINE_KEYWORDS if word.lower() in str(text).lower()) for text in texts]
    features['has_first_person'] = [int(bool(re.search(r"\b(i|we|my|our)\b", str(text).lower()))) for text in texts]
    features['has_past_tense'] = [int(bool(re.search(r"\b(was|were|went|visited|ate|had|ordered)\b", str(text).lower()))) for text in texts]
    features['first_person_past_combo'] = [int(fp and pt) for fp, pt in zip(features['has_first_person'], features['has_past_tense'])]
    
    # RANT features  
    features['rant_word_count'] = [sum(1 for word in NEVER_VISITED_PHRASES if word.lower() in str(text).lower()) for text in texts]
    features['strong_rant_count'] = [sum(1 for word in STRONG_RANT_WORDS if word.lower() in str(text).lower()) for text in texts]
    features['never_visited_explicit'] = [int(bool(NEVER_EXPLICIT_RX.search(str(text)))) for text in texts]
    features['hearsay_pattern'] = [int(bool(HEARSAY_RX.search(str(text)))) for text in texts]
    
    # IRRELEVANT features
    features['irrelevant_word_count'] = [sum(1 for word in IRRELEVANT_KEYWORDS if word.lower() in str(text).lower()) for text in texts]
    features['has_politics'] = [int(bool(re.search(r"\b(election|president|congress|parliament|democrat|republican|politics)\b", str(text).lower()))) for text in texts]
    features['has_employment'] = [int(bool(re.search(r"\b(hiring|apply|resume|vacancy|job\s+opening|career|recruit|salary)\b", str(text).lower()))) for text in texts]
    features['generic_template'] = [int(bool(re.match(r"^(nice|great|good|amazing)\s+(place|food|service)$", str(text).lower().strip()))) for text in texts]
    
    return features

def extract_domain_features(df):
    """Extract domain-specific features based on business category"""
    features = {'domain_hit_ratio': [], 'off_domain_ratio': []}
    
    for _, row in df.iterrows():
        text = str(row.get('text_clean', '')).lower()
        cat = str(row.get('category_primary', '') or row.get('category_all', ''))
        
        try:
            fam = category_to_family(cat)
        except:
            fam = "generic"
            
        if fam == "generic":
            features['domain_hit_ratio'].append(0.0)
            features['off_domain_ratio'].append(0.0)
            continue
            
        toks = re.findall(r"\b[a-z]+\b", text)
        if not toks:
            features['domain_hit_ratio'].append(0.0)
            features['off_domain_ratio'].append(0.0)
            continue
            
        try:
            dom = DOMAIN_TERMS.get(fam, set())
            toks_filtered = [t for t in toks if t not in WHITELIST_COMMON]
            
            if not toks_filtered:
                features['domain_hit_ratio'].append(0.0)
                features['off_domain_ratio'].append(0.0)
                continue
                
            domain_hits = sum(1 for t in toks_filtered if t in dom)
            off_hits = sum(1 for t in toks_filtered if t not in dom)
            
            features['domain_hit_ratio'].append(domain_hits / len(toks_filtered))
            features['off_domain_ratio'].append(off_hits / len(toks_filtered))
        except:
            features['domain_hit_ratio'].append(0.0)
            features['off_domain_ratio'].append(0.0)
    
    return features

def extract_tfidf_features(texts, keyword_sets, max_features=50):
    """Extract TF-IDF features for different keyword categories"""
    tfidf_features = {}
    
    for category, keywords in keyword_sets.items():
        if not keywords:
            continue
            
        vectorizer = TfidfVectorizer(
            vocabulary=list(keywords),
            ngram_range=(1, 2),
            max_features=max_features
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([str(text) for text in texts])
            feature_names = [f"{category}_tfidf_{name}" for name in vectorizer.get_feature_names_out()]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
            tfidf_features.update(tfidf_df.to_dict('list'))
        except:
            # Handle case where no keywords are found
            for kw in list(keywords)[:max_features]:
                tfidf_features[f"{category}_tfidf_{kw}"] = [0.0] * len(texts)
    
    return tfidf_features

def extract_embedding_features(texts, batch_size=64):
    """Extract sentence embedding features"""
    embeddings = sbert_model.encode([str(text) for text in texts], 
                                   batch_size=batch_size, 
                                   show_progress_bar=True)
    
    # Create feature names
    feature_names = [f"embedding_{i}" for i in range(embeddings.shape[1])]
    
    return {name: embeddings[:, i].tolist() for i, name in enumerate(feature_names)}

def extract_metadata_features(df):
    """Extract features from review metadata"""
    features = {}
    
    # Rating-based features
    if 'rating' in df.columns:
        features['rating'] = df['rating'].fillna(0).tolist()
        features['high_rating'] = (df['rating'] >= 4).astype(int).tolist()
        features['low_rating'] = (df['rating'] <= 2).astype(int).tolist()
    
    # Photo-based features
    if 'has_photos' in df.columns:
        features['has_photos'] = df['has_photos'].fillna(0).astype(int).tolist()
    if 'photo_count' in df.columns:
        features['photo_count'] = df['photo_count'].fillna(0).tolist()
        features['multiple_photos'] = (df['photo_count'] > 1).astype(int).tolist()
    
    # Response features
    if 'has_response' in df.columns:
        features['has_response'] = df['has_response'].fillna(0).astype(int).tolist()
    
    return features

def create_interaction_features(feature_dict):
    """Create interaction features between different categories"""
    interactions = {}
    
    # Interaction between rating and text length
    if 'rating' in feature_dict and 'word_count' in feature_dict:
        interactions['rating_length_interaction'] = [
            r * wc for r, wc in zip(feature_dict['rating'], feature_dict['word_count'])
        ]
    
    # Interaction between first person and domain relevance
    if 'has_first_person' in feature_dict and 'domain_hit_ratio' in feature_dict:
        interactions['first_person_domain_interaction'] = [
            fp * dhr for fp, dhr in zip(feature_dict['has_first_person'], feature_dict['domain_hit_ratio'])
        ]
    
    # Interaction between photos and word count
    if 'has_photos' in feature_dict and 'word_count' in feature_dict:
        interactions['photos_length_interaction'] = [
            hp * wc for hp, wc in zip(feature_dict['has_photos'], feature_dict['word_count'])
        ]
    
    return interactions

def build_comprehensive_features(file_path, include_embeddings=True):
    """Build comprehensive feature set for all label categories"""
    print("Loading data...")
    df = load_cleaned_csv(file_path)
    texts = df["text_clean"].tolist()
    
    # Initialize feature dictionary
    all_features = {'text_clean': texts}
    
    # Copy relevant original columns
    for col in ['rating', 'has_photos', 'photo_count', 'has_response', 'category_primary', 'category_all']:
        if col in df.columns:
            all_features[col] = df[col].tolist()
    
    # Extract basic features
    print("Extracting basic features...")
    basic_features = extract_basic_features(texts)
    all_features.update(basic_features)
    
    # Extract POS features
    print("Extracting POS features...")
    pos_features = extract_pos_features(texts)
    all_features.update(pos_features)
    
    # Extract keyword features
    print("Extracting keyword features...")
    keyword_features = extract_keyword_features(texts)
    all_features.update(keyword_features)
    
    # Extract domain features
    print("Extracting domain features...")
    domain_features = extract_domain_features(df)
    all_features.update(domain_features)
    
    # Extract metadata features
    print("Extracting metadata features...")
    metadata_features = extract_metadata_features(df)
    all_features.update(metadata_features)
    
    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    keyword_sets = {
        'promotion': PROMO_WORDS,
        'genuine': GENUINE_KEYWORDS,
        'rant': NEVER_VISITED_PHRASES | STRONG_RANT_WORDS,
        'irrelevant': IRRELEVANT_KEYWORDS
    }
    tfidf_features = extract_tfidf_features(texts, keyword_sets)
    all_features.update(tfidf_features)
    
    # Extract embedding features (optional due to computational cost)
    if include_embeddings:
        print("Extracting embedding features...")
        embedding_features = extract_embedding_features(texts)
        all_features.update(embedding_features)
    
    # Create interaction features
    print("Creating interaction features...")
    interaction_features = create_interaction_features(all_features)
    all_features.update(interaction_features)
    
    # Create composite flags for each label category
    print("Creating composite label flags...")
    n_samples = len(texts)
    
    # PROMOTION flags
    all_features['promotion_flag'] = [
        int(any([
            all_features['has_url'][i],
            all_features['has_email'][i], 
            all_features['has_phone'][i],
            all_features['promo_word_count'][i] > 0,
            all_features['has_social_media'][i]
        ])) for i in range(n_samples)
    ]
    
    # GENUINE flags
    all_features['genuine_flag'] = [
        int(any([
            all_features['first_person_past_combo'][i],
            all_features['genuine_word_count'][i] >= 2,
            all_features['domain_hit_ratio'][i] > 0.2,
            (all_features.get('high_rating', [0]*n_samples)[i] and all_features['word_count'][i] >= 10)
        ])) for i in range(n_samples)
    ]
    
    # RANT flags
    all_features['rant_flag'] = [
        int(any([
            all_features['never_visited_explicit'][i],
            all_features['hearsay_pattern'][i],
            all_features['strong_rant_count'][i] > 0,
            all_features['rant_word_count'][i] > 0
        ])) for i in range(n_samples)
    ]
    
    # IRRELEVANT flags
    all_features['irrelevant_flag'] = [
        int(any([
            all_features['has_politics'][i],
            all_features['has_employment'][i],
            all_features['generic_template'][i],
            all_features['off_domain_ratio'][i] > 0.8,
            all_features['word_count'][i] < 3
        ])) for i in range(n_samples)
    ]
    
    # Overall quality score
    all_features['quality_score'] = [
        all_features['genuine_flag'][i] * 2 + 
        all_features['domain_hit_ratio'][i] * 3 +
        (1 if all_features['word_count'][i] >= 10 else 0) +
        all_features.get('has_photos', [0]*n_samples)[i] -
        all_features['promotion_flag'][i] * 2 -
        all_features['rant_flag'][i] -
        all_features['irrelevant_flag'][i]
        for i in range(n_samples)
    ]
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    print(f"Feature extraction complete! Generated {len(features_df.columns)} features for {len(features_df)} samples.")
    return features_df

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    # Build comprehensive features
    features_df = build_comprehensive_features(
        "googlelocal_reviews_cleaned.csv", 
        include_embeddings=True  # Set to False for faster processing
    )
    
    # Save results
    output_file = "googlelocal_review_features_comprehensive.csv"
    features_df.to_csv(output_file, index=False)
    
    print(f"\nFeature extraction complete!")
    print(f"Saved {len(features_df)} samples with {len(features_df.columns)} features to {output_file}")
    print("\nFeature summary:")
    print(f"- Basic features: character/word counts, ratios")
    print(f"- POS features: noun, adjective, verb, adverb counts")
    print(f"- Keyword features: category-specific word counts")
    print(f"- Domain features: business category relevance")
    print(f"- TF-IDF features: weighted term frequencies")
    print(f"- Embedding features: sentence-level representations")
    print(f"- Metadata features: ratings, photos, responses")
    print(f"- Interaction features: cross-feature combinations")
    print(f"- Label flags: composite indicators for each category")
    
    # Display feature distribution
    print(f"\nLabel flag distributions:")
    for flag in ['promotion_flag', 'genuine_flag', 'rant_flag', 'irrelevant_flag']:
        if flag in features_df.columns:
            count = features_df[flag].sum()
            pct = (count / len(features_df)) * 100
            print(f"- {flag}: {count} samples ({pct:.1f}%)")