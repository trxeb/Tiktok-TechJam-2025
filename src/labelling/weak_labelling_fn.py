import pandas as pd
import matplotlib.pyplot as plt
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Keyword lists for heuristics
PROMO_KEYWORDS = [
    "discount", "sale", "buy now", "limited offer", "promotion",
    "deal", "coupon", "special offer", "order online"
]

SPAM_KEYWORDS = [
    "free", "click here", "subscribe", "visit my", "check out",
    "buy this", "amazing product", "best ever", "unrelated"
]

NEGATIVE_INDICATORS = [
    "heard", "supposedly", "people say", "rumor", "never visited",
    "not been", "didn't go"
]

FIRST_PERSON_VISIT_VERBS = [
    "went", "visited", "tried", "ordered", "ate", "experienced"
]

# -------------------------------
# Topic relevance check


def topic_relevance(text_clean: str, category: str, threshold: float = 0.4) -> bool:
    """
    Returns True if review is relevant to the business_category.
    Uses spaCy similarity between review text and category phrase.
    """
    if not isinstance(text_clean, str) or not text_clean.strip():
        return False
    if not isinstance(category, str) or not category.strip():
        return True  # treat as relevant if no category info

    doc_review = nlp(text_clean)
    doc_category = nlp(category)

    similarity = doc_review.similarity(doc_category)
    return similarity >= threshold

# -------------------------------
# Improved weak labeling function


def weak_label_review_improved(text: str, category: str) -> str:
    """
    Returns coarse weak label: Likely Genuine, Likely Promotional,
    Non-genuine / spam / irrelevant / rants.
    Considers:
    - Keyword heuristics
    - NLP dependency parsing
    - Topic relevance with business category
    """
    if not isinstance(text, str) or not text.strip():
        return "Non-genuine / spam / irrelevant / rants"

    text_lower = text.lower()

    # 1. Promotional
    if any(keyword in text_lower for keyword in PROMO_KEYWORDS):
        return "Likely Promotional"

    # 2. Spam keywords
    if any(keyword in text_lower for keyword in SPAM_KEYWORDS):
        return "Non-genuine / spam / irrelevant / rants"

    # 3. NLP-based
    doc = nlp(text)
    first_person_experience = False
    negation = False
    second_hand = False

    for token in doc:
        if token.dep_ == "neg":
            negation = True
        if token.lemma_ in FIRST_PERSON_VISIT_VERBS and any(child.dep_ == "nsubj" and child.text.lower() == "i" for child in token.children):
            first_person_experience = True
        if any(kw in text_lower for kw in NEGATIVE_INDICATORS):
            second_hand = True

    if first_person_experience and not negation:
        base_label = "Likely Genuine"
    elif second_hand or (not first_person_experience and negation):
        base_label = "Non-genuine / spam / irrelevant / rants"
    elif len(text.split()) < 3:
        base_label = "Non-genuine / spam / irrelevant / rants"
    else:
        base_label = "Likely Genuine"

    # 4. Topic relevance adjustment
    relevant = topic_relevance(text, category)
    if not relevant and base_label == "Likely Genuine":
        return "Non-genuine / spam / irrelevant / rants"

    return base_label


# -------------------------------
# Load your CSV
# replace with your file
input_csv = "/Users/peienlee/Documents/GitHub/Tiktok-TechJam-2025/cleaned_with_category.csv"
df = pd.read_csv(input_csv)

# Apply weak labeling
for i, row in enumerate(df.itertuples()):
    review_text = row.text_clean
    category = row.category  # fetch the category for this review
    df.at[i, 'weak_label'] = weak_label_review_improved(review_text, category)

    if i % 100 == 0:
        print(f"Processed {i} reviews...")

# Save weak-labeled CSV
output_csv = "weak_labeled_reviews.csv"
df.to_csv(output_csv, index=False)
print(f"Weak-labeled CSV saved to {output_csv}")

# -------------------------------
# Preview first 15 rows
print("\nPreview of first 15 rows:")
print(df.head(15))

# -------------------------------
# Plot label distribution
label_counts = df['weak_label'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(label_counts.index, label_counts.values,
        color=['green', 'orange', 'red'])
plt.title("Weak Label Distribution")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=20)
plt.show()
