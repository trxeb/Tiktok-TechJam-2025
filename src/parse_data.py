# read line by line 
import gzip
import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import scipy.sparse as sp


def parse(path):
    open_func = gzip.open if str(path).endswith('.gz') else open
    mode = 'rt' if open_func is gzip.open else 'r'
    with open_func(path, mode, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

import pandas as pd
from parse_data import parse  # your parse() function

# -------------------------------
# Train on Alaska (same as before)
# -------------------------------
reviews_alaska = list(parse("data/review-Alaska_10.json"))
df_alaska = pd.DataFrame(reviews_alaska)

df_alaska['sentiment'] = df_alaska['rating'].apply(lambda x: 1 if x > 3 else 0)
df_ml_alaska = df_alaska[['text', 'sentiment']].copy()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_ml_alaska['clean_text'] = df_ml_alaska['text'].apply(clean_text)

# Heuristic features
df_ml_alaska['review_length'] = df_ml_alaska['clean_text'].apply(lambda x: len(x.split()))
df_ml_alaska['has_url'] = df_ml_alaska['text'].apply(lambda x: int('http' in str(x) or 'www.' in str(x)))
keywords = ['never visited', 'did not go', 'not there', 'didn’t visit']
df_ml_alaska['likely_rant'] = df_ml_alaska['clean_text'].apply(lambda x: int(any(k in x for k in keywords)))

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_text_alaska = vectorizer.fit_transform(df_ml_alaska['clean_text'])

X_meta_alaska = df_ml_alaska[['review_length', 'has_url', 'likely_rant']].values
X_alaska = sp.hstack([X_text_alaska, X_meta_alaska])

df_ml_alaska['low_quality'] = ((df_ml_alaska['has_url'] == 1) |
                               (df_ml_alaska['likely_rant'] == 1) |
                               (df_ml_alaska['review_length'] < 5)).astype(int)
y_alaska = df_ml_alaska['low_quality'].values

# Save only the cleaned version and relevant columns
df_cleaned = df_ml_alaska[['clean_text', 'sentiment', 'review_length', 'has_url', 'likely_rant', 'low_quality']]
df_cleaned.to_csv("cleaned_alaska.csv", index=False)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_alaska, y_alaska)


# -------------------------------
# Test on DC dataset
# -------------------------------
reviews_dc = list(parse("data/review-District_of_Columbia_10.json"))
df_dc = pd.DataFrame(reviews_dc)

df_dc['sentiment'] = df_dc['rating'].apply(lambda x: 1 if x > 3 else 0)
df_ml_dc = df_dc[['text', 'sentiment']].copy()
df_ml_dc['clean_text'] = df_ml_dc['text'].apply(clean_text)

# Heuristic features for DC
df_ml_dc['review_length'] = df_ml_dc['clean_text'].apply(lambda x: len(x.split()))
df_ml_dc['has_url'] = df_ml_dc['text'].apply(lambda x: int('http' in str(x) or 'www.' in str(x)))
df_ml_dc['likely_rant'] = df_ml_dc['clean_text'].apply(lambda x: int(any(k in x for k in keywords)))

# Transform DC text using the Alaska TF-IDF vectorizer
X_text_dc = vectorizer.transform(df_ml_dc['clean_text'])
X_meta_dc = df_ml_dc[['review_length', 'has_url', 'likely_rant']].values
X_dc = sp.hstack([X_text_dc, X_meta_dc])

df_ml_dc['low_quality'] = ((df_ml_dc['has_url'] == 1) |
                           (df_ml_dc['likely_rant'] == 1) |
                           (df_ml_dc['review_length'] < 5)).astype(int)
y_dc = df_ml_dc['low_quality'].values

# Evaluate on DC
y_pred_dc = model.predict(X_dc)
print("Accuracy on DC dataset:", accuracy_score(y_dc, y_pred_dc))
print(classification_report(y_dc, y_pred_dc))
