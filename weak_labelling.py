import re
import pandas as pd
from typing import Optional

# Define your keyword lists
PROMO_WORDS = {'discount', 'free', 'promo', 'coupon', 'code', 'deal', 'offer', 'save $', '% off'}
HEARSAY_WORDS = {'heard', 'looks', 'seems', 'probably', 'think', 'might be', 'looked', 'sounded'}
NEVER_VISITED_PHRASES = {
    "never visited", "never been", "haven't been", "haven't visited", "didn't go",
    "not been", "not go", "hasn't been", "hasn't visited", "never actually went"
}

def lf_contains_url(text: str) -> Optional[int]:
    """Label 0 if review contains a URL."""
    pattern = r'https?://\S+|www\.\S+'
    return 0 if re.search(pattern, text) is not None else -1

def lf_contains_promo_words(text: str) -> Optional[int]:
    """Label 0 if review contains promotional words."""
    text_lower = text.lower()
    return 0 if any(word in text_lower for word in PROMO_WORDS) else -1

def lf_never_visited(text: str) -> Optional[int]:
    """Label 0 if review states user never visited."""
    text_lower = text.lower()
    return 0 if any(phrase in text_lower for phrase in NEVER_VISITED_PHRASES) else -1

def lf_high_quality_signal(text: str) -> Optional[int]:
    """Label 1 if review has strong signals of being genuine (first-person, past-tense)."""
    first_person = {' i ', ' me ', ' my ', ' we ', ' our '}
    past_tense = {' was ', ' were ', ' did ', ' had ', ' ordered ', ' visited ', ' went ', ' came ', ' ate ', ' paid '}
    text_lower = f" {text.lower()} " # Add spaces for whole word matching

    if any(word in text_lower for word in first_person) and any(word in text_lower for word in past_tense):
        return 1
    else:
        return -1

def lf_all_caps_ratio(text: str) -> Optional[int]:
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return -1
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
    ratio = caps_count / len(words)
    return 0 if ratio > 0.3 else -1

def lf_competitor_mention(text: str) -> Optional[int]:
    if re.search(r'\b(vs|compared to)\b', text.lower()):
        if re.search(r'\b[A-Z][a-z]+\b', text):  # crude check for proper names
            return 0
    return -1

def lf_generic_short_review(text: str) -> Optional[int]:
    words = text.strip().split()
    if len(words) < 4:
        return 0
    generic_phrases = {"good place", "nice food", "great service", "amazing place"}
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in generic_phrases):
        return 0
    return -1

def lf_too_long(text: str) -> Optional[int]:
    return 0 if len(text) > 1000 else -1
