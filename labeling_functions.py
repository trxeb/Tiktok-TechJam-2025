# labeling_functions.py
import pandas as pd
import re
from snorkel.labeling import labeling_function

# ----------------------------
# Define Constants for Labels
# ----------------------------
ABSTAIN = -1
GENUINE = 0
PROMOTION = 1
IRRELEVANT = 2
RANT = 3

# Define our key word sets
PROMO_WORDS = {
    'discount', 'free', 'promo', 'coupon', 'code', 'deal', 'offer', 'save $', '% off', 
    'buy', 'purchase', 'sale', 'promotion', 'promotional', 'special offer', 'limited time', 
    'best price', 'cheap', 'cheapest', 'cheap deal', 'visit our website', 'follow us', 
    'subscribe', 'contact us', 'call now', 'sign up', 'join now', 'check out', 'shop now'
    'recommended by', 'must-have'}
NEVER_VISITED_PHRASES = {
    "never visited", "never been", "haven't been", "haven't visited", "didn't go",
    "not been", "not go", "hasn't been", "hasn't visited", "never actually went",
    'not visited', 'someone told me', 'I read that', 'was not there' 
    'heard', 'seems', 'probably', 'think', 'might', 'looks', 
    "terrible", "worst", "awful", "scam", "cheated", "fraud","disgusting", "rude", "never again"
}

# ----------------------------
# Labeling Functions
# Each function now accepts a row and extracts the text from 'text_clean' column
# ----------------------------

@labeling_function()
def lf_contains_promo_words(row):
    text = str(row['text_clean']).lower()
    if any(word in text for word in PROMO_WORDS):
        return PROMOTION
    return ABSTAIN

@labeling_function()
def lf_never_visited(row):
    text = str(row['text_clean']).lower()
    if any(phrase in text for phrase in NEVER_VISITED_PHRASES):
        return RANT
    return ABSTAIN

@labeling_function()
def lf_high_quality_signal(row):
    text = str(row['text_clean']).lower()
    first_person = {' i ', ' me ', ' my ', ' we ', ' our '}
    past_tense = {' was ', ' were ', ' did ', ' had ', ' ordered ',
                  ' visited ', ' went ', ' came ', ' ate ', ' paid '}
    # Add spaces to avoid matching parts of words (e.g., "menu")
    text_with_spaces = f" {text} "
    if any(w in text_with_spaces for w in first_person) and any(w in text_with_spaces for w in past_tense):
        return GENUINE
    return ABSTAIN

@labeling_function()
def lf_generic_short_review(row):
    text = str(row['text_clean'])
    words = text.strip().split()
    if len(words) < 2:
        return IRRELEVANT
    generic_phrases = {"good place", "nice food", "great service", "amazing place"}
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in generic_phrases):
        return IRRELEVANT
    return ABSTAIN

@labeling_function()
def lf_contains_url_phone_email(row):
    text = str(row['text_clean']).lower()
    # URL pattern
    url_pattern = r'(http[s]?://|www\.)\S+'
    # Email pattern
    email_pattern = r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b'
    # Phone pattern (simple, matches numbers with optional +, -, (), spaces)
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}'
    if re.search(url_pattern, text) or re.search(email_pattern, text) or re.search(phone_pattern, text):
        return PROMOTION
    return ABSTAIN

@labeling_function()
def lf_long_review(row):
    if len(str(row['text_clean']).split()) > 50:
        return RANT
    return ABSTAIN

# ----------------------------
# Create a list of all LFs to use later
# ----------------------------
lfs = [
    lf_contains_promo_words,
    lf_never_visited,
    lf_high_quality_signal,
    lf_generic_short_review,
    lf_contains_url_phone_email,
    lf_long_review
]