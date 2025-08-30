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

_TEXT_COL = "text_clean"
_CATEGORY_COL = "category_primary"

def set_text_col(col: str):
    global _TEXT_COL
    _TEXT_COL = col

def set_category_col(col: str):
    global _CATEGORY_COL
    _CATEGORY_COL = col

# Define our key word sets
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
    "not visited", "someone told me", "i read that", "was not there", "i heard",
    "seems", "probably", "think", "might", "looks"
}

STRONG_RANT_WORDS = {
    "terrible", "worst", "awful", "scam", "cheated", "fraud", "disgusting", "rude", "never again"
}


#added functions to use category 
# Map raw meta categories to a coarse family using substring checks (lowercased).
FAMILY_MAP = {
    # Food & Drink
    "restaurant": {
        "restaurant", "fast food", "bar & grill", "grill", "steakhouse", "bbq", "pizzeria",
        "pizza", "sandwich shop", "burger", "mexican restaurant", "american restaurant",
        "breakfast restaurant", "takeout restaurant", "ice cream", "diner", "seafood",
        "sushi", "ramen", "noodle", "buffet", "curry", "taco", "wing", "bistro"
    },
    "cafe_coffee_bakery": {
        "coffee", "coffee shop", "cafe", "tea", "bubble tea", "bakery", "donut", "pastry",
        "dessert", "bagel"
    },
    "bar_pub": {
        "bar", "pub", "taproom", "brewery", "distillery", "wine bar"
    },

    # Lodging / Outdoors / Attractions
    "hotel_lodging": {
        "hotel", "motel", "lodging", "resort", "inn", "hostel"
    },
    "campground_rv": {
        "campground", "rv", "rv park", "trailer park"
    },
    "attraction_park": {
        "tourist attraction", "park", "trail", "hiking", "monument", "museum", "zoo",
        "aquarium", "art gallery", "theater", "amusement", "river", "lake", "hot spring",
        "national park", "state park", "scenic", "viewpoint"
    },

    # Retail (general + specific)
    "grocery_convenience": {
        "grocery store", "supermarket", "convenience store", "market", "liquor store",
        "butcher", "fishmonger", "deli"
    },
    "retail_general": {
        "store", "shopping", "gift shop", "thrift", "consignment", "flea market"
    },
    "retail_clothing": {
        "clothing store", "apparel", "boutique", "shoe", "jewelry"
    },
    "retail_specialty": {
        "electronics", "hardware", "furniture", "book", "pet store", "sporting goods",
        "outdoor store", "auto parts store", "tire shop", "pharmacy"
    },

    # Auto, Gas, Transport
    "auto_service": {
        "auto repair", "car repair", "auto service", "oil change", "tire shop", "body shop",
        "car wash", "car detailing"
    },
    "fuel_station": {
        "gas station", "fuel", "truck stop"
    },
    "transport": {
        "airport", "bus station", "train station", "taxi", "rideshare"
    },

    # Health, Beauty, Wellness
    "healthcare": {
        "clinic", "hospital", "urgent care", "pharmacy", "dentist", "chiropractor",
        "optometrist", "veterinary", "rehabilitation", "wellness", "holistic", "alternative medicine"
    },
    "beauty_wellness": {
        "beauty salon", "hair salon", "barber", "spa", "nail", "massage", "tattoo", "piercing"
    },

    # Services / Offices
    "real_estate": {
        "real estate", "property management", "apartment", "housing"
    },
    "professional_services": {
        "lawyer", "attorney", "accountant", "insurance", "consultant", "marketing",
        "photography", "printing", "copy center", "it service", "computer repair"
    },
    "finance": {
        "bank", "credit union", "atm", "loan", "financial"
    },
    "education": {
        "school", "high school", "college", "university", "library", "child care", "daycare"
    },
    "religious": {
        "church", "temple", "mosque", "synagogue"
    },
    "government_postal": {
        "city hall", "county", "government", "dmv", "post office", "courthouse", "fire station",
        "police", "sheriff"
    },
}

DOMAIN_TERMS = {
    "restaurant": {
        "food","menu","dish","meal","taste","flavor","service","staff","server","waiter","waitress",
        "drink","coffee","tea","soda","beer","wine","cocktail","portion","price","bill","order",
        "table","seat","reservation","ambience","ambiance","atmosphere","spicy","crispy","fresh",
        "sauce","salad","appetizer","entree","dessert","breakfast","lunch","dinner","queue","takeout","delivery"
    },
    "cafe_coffee_bakery": {
        "coffee","latte","espresso","americano","cappuccino","mocha","bean","brew","roast","barista",
        "tea","milk tea","boba","bubble","pastry","cake","cookie","muffin","bagel","bread","croissant",
        "dessert","sweet","sugar","bakery","counter","cup","mug","foam","crema"
    },
    "bar_pub": {
        "beer","tap","pint","ale","lager","ipa","stout","wine","whiskey","bourbon","vodka","rum",
        "cocktail","bar","happy hour","bartender","pub","shots","craft","brews"
    },
    "hotel_lodging": {
        "room","bed","king","queen","suite","stay","check-in","checkin","checkout","front desk",
        "lobby","housekeeping","clean","towel","amenities","breakfast","wifi","parking","quiet",
        "noise","view","elevator","ac","heater","pool","gym","spa","resort"
    },
    "campground_rv": {
        "camp","camping","tent","rv","hookup","site","fire pit","picnic","trail","hike","bathhouse",
        "restroom","nature","wildlife","shade","spot","reservation","check-in"
    },
    "attraction_park": {
        "trail","hike","scenic","view","overlook","waterfall","river","lake","wildlife","museum",
        "exhibit","gallery","show","performance","theater","park","picnic","playground","entrance","fee"
    },
    "grocery_convenience": {
        "grocery","produce","vegetable","fruit","meat","dairy","milk","bread","egg","aisle","shelf",
        "checkout","cashier","receipt","cart","basket","price","deal","sale","stock","inventory"
    },
    "retail_general": {
        "store","cashier","checkout","receipt","return","refund","stock","aisle","price","sale","deal"
    },
    "retail_clothing": {
        "clothing","apparel","shirt","pants","jeans","dress","shoe","size","fit","try on","fitting",
        "style","fashion","fabric","return","exchange"
    },
    "retail_specialty": {
        "electronics","hardware","tools","laptop","phone","printer","paint","nail","screw","pet food",
        "leash","pharmacy","medication","prescription","tire","oil","filter","book","novel"
    },
    "auto_service": {
        "car","auto","vehicle","repair","service","maintenance","oil change","brake","tire","alignment",
        "inspection","diagnostic","mechanic","appointment","quote","warranty","shop","garage"
    },
    "fuel_station": {
        "gas","fuel","pump","diesel","octane","station","price","pay","cashier","restroom","convenience"
    },
    "transport": {
        "airport","flight","terminal","gate","boarding","bus","route","schedule","ticket","train",
        "platform","taxi","pickup","dropoff"
    },
    "healthcare": {
        "clinic","doctor","physician","nurse","appointment","checkup","diagnosis","treatment","prescription",
        "pharmacy","medicine","therapy","rehab","chiropractor","x-ray","insurance","copay","waiting room"
    },
    "beauty_wellness": {
        "salon","stylist","haircut","color","perm","nails","manicure","pedicure","massage","facial",
        "spa","wax","barber","shave","appointment","booking"
    },
    "real_estate": {
        "real estate","agent","realtor","listing","showing","closing","mortgage","lease","rent","tenant",
        "property","inspection","offer","escrow","hoa"
    },
    "professional_services": {
        "lawyer","attorney","case","court","contract","accountant","tax","audit","consult","it support",
        "network","backup","marketing","campaign","project","quote","invoice"
    },
    "finance": {
        "bank","branch","teller","atm","loan","credit","debit","account","deposit","withdraw",
        "interest","fee","transfer","mortgage"
    },
    "education": {
        "school","teacher","class","student","lesson","course","library","study","homework","campus"
    },
    "religious": {
        "church","mass","service","pastor","sermon","worship","congregation","temple","mosque","synagogue"
    },
    "government_postal": {
        "city hall","county","government","permit","license","clerk","post office","mail","package",
        "shipping","passport","court","dmv","sheriff","police","fire station"
    },
    "generic": set()
}

WHITELIST_COMMON = {
    "good","great","nice","amazing","awesome","love","like","service","place",
    "staff","people","time","today","yesterday","experience","best","better",
    "bad","okay","ok","friendly","helpful","clean","dirty","fast","slow"
}

# --- Never-visited / hearsay patterns (explicit & implicit) ---
NEVER_EXPLICIT_RX = re.compile(
    r"(never\s+(been|visited)|haven'?t\s+(been|visited)|didn'?t\s+(go|visit)|"
    r"\bnot\s+(been|go|visit)\b|was(n't| not)\s+there|"
    r"\bnever\s+actually\s+(went|visited)\b)",
    re.I
)

HEARSAY_RX = re.compile(
    r"\b(i\s+heard|someone\s+said|people\s+say|they\s+say|"
    r"i\s+read\s+(online|about)|rumor(ed)?)\b", re.I
)

DRIVEBY_RX = re.compile(
    r"\b(drove\s+by|drove\s+past|passed\s+by|just\s+saw\s+it)\b", re.I
)

PHONE_ONLY_RX = re.compile(
    r"\b(called|phoned|on\s+the\s+phone|left\s+a\s+message|voicemail|"
    r"no\s+answer|line\s+was\s+busy)\b", re.I
)

CLOSED_NO_ENTRY_RX = re.compile(
    r"\b(closed\s+when\s+we\s+arrived|could(n't| not)\s+get\s+in|"
    r"no\s+reservation|turn(ed)?\s+away)\b", re.I
)

# --- Off-topic domains you likely don't want in place reviews ---
POLITICS_RX = re.compile(r"\b(election|president|congress|parliament|democrat|republican|left|right|policy|politics)\b", re.I)
EMPLOYMENT_RX = re.compile(r"\b(hiring|apply|resume|vacancy|job\s+opening|career|recruit|salary|benefit)\b", re.I)
GENERIC_TEMPLATE_RX = re.compile(
    r"^(nice|great|good|amazing)\s+(place|food|service)$", re.I
)


def category_to_family(cat_raw: str) -> str:
    """
    Map a raw category string (e.g., 'Fast food restaurant;Restaurant')
    to one of the FAMILY_MAP keys above.
    """
    if not cat_raw:
        return "generic"
    low = str(cat_raw).lower()
    for fam, keys in FAMILY_MAP.items():
        if any(k in low for k in keys):
            return fam
    return "generic"

# ----------------------------
# Labeling Functions
# Each function now accepts a row and extracts the text from 'text_clean' column
# ----------------------------

@labeling_function()
def lf_contains_promo_words(row):
    text = str(row.get('text_clean', '')).lower()
    if any(word in text for word in PROMO_WORDS):
        return PROMOTION
    return ABSTAIN

@labeling_function()
def lf_never_visited(row):
    text = str(row.get('text_clean', '')).lower()
    if any(phrase in text for phrase in NEVER_VISITED_PHRASES):
        return RANT
    return ABSTAIN

@labeling_function()
def lf_high_quality_signal(row):
    text = str(row.get('text_clean', '')).lower()
    first_person = {' i ', ' me ', ' my ', ' we ', ' our '}
    past_tense = {' was ', ' were ', ' did ', ' had ', ' ordered ',
                  ' visited ', ' went ', ' came ', ' ate ', ' paid '}
    padded = f" {text} "
    if any(w in padded for w in first_person) and any(w in padded for w in past_tense):
        return GENUINE
    return ABSTAIN

@labeling_function()
def lf_generic_short_review(row):
    text = str(row.get('text_clean', '') or '')
    text_lower = text.lower().strip()
    wc = len(re.findall(r"\b\w+\b", text_lower))
    generic_phrases = {"good place", "nice food", "great service", "amazing place"}

    # If it's short and generic but rated well, treat as genuine instead of irrelevant
    rating = row.get('rating', None)
    if wc <= 6 and any(p in text_lower for p in generic_phrases):
        try:
            if rating is not None and float(rating) >= 4:
                return GENUINE
        except:
            pass
        return IRRELEVANT

    return ABSTAIN


@labeling_function()
def lf_contains_url_phone_email(row):
    text = str(row.get('text_clean', '')).lower()
    url_pattern = r'(http[s]?://|www\.)\S+'
    email_pattern = r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b'
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}'
    if re.search(url_pattern, text) or re.search(email_pattern, text) or re.search(phone_pattern, text):
        return PROMOTION
    return ABSTAIN

@labeling_function()
def lf_offdomain_by_category(row):
    text = str(row.get('text_clean', '') or '').lower()
    cat  = str(row.get('category_primary', '') or row.get('category_all',''))
    fam = category_to_family(cat)
    if fam == "generic":
        return ABSTAIN

    toks = re.findall(r"\b[a-z]+\b", text)
    wc = len(toks)
    # require enough text to judge
    if wc < 25:
        return ABSTAIN

    dom = DOMAIN_TERMS.get(fam, set())
    # don’t penalize common adjectives/adverbs
    toks_filtered = [w for w in toks if w not in WHITELIST_COMMON]

    domain_hits = sum(1 for w in toks_filtered if w in dom)
    off_hits    = sum(1 for w in toks_filtered if w not in dom)
    tot = domain_hits + off_hits
    if tot == 0:
        return ABSTAIN

    off_ratio = off_hits / tot
    dom_ratio = domain_hits / tot

    # if there's reasonable on-topic evidence, abstain
    if domain_hits >= 2 or dom_ratio >= 0.12:
        return ABSTAIN

    # only call IRRELEVANT when very off-domain
    return IRRELEVANT if off_ratio > 0.90 else ABSTAIN


# ----- NEW POSITIVE LFs (return GENUINE, not 1) -----

@labeling_function()
def lf_genuine_first_person(row):
    text = " " + str(row.get('text_clean', '')).lower() + " "
    if any(p in text for p in [" i ", " we ", " my ", " our "]) and \
       any(v in text for v in [" was ", " went ", " had ", " ordered ", " ate ", " visited "]):
        return GENUINE
    return ABSTAIN

@labeling_function()
def lf_genuine_descriptive(row):
    text = str(row.get('text_clean', ''))
    wc = len(re.findall(r"\b\w+\b", text))
    padded = " " + text.lower() + " "
    if wc >= 12 and any(adj in padded for adj in [" delicious ", " crispy ", " spicy ", " sweet ",
                                                  " salty ", " creamy ", " crunchy ", " tender ",
                                                  " juicy ", " friendly staff ", " clean room ", " comfortable bed "]):
        return GENUINE
    return ABSTAIN

@labeling_function()
def lf_rant_never_visit_explicit(row):
    t = str(row.get('text_clean','') or '')
    return RANT if NEVER_EXPLICIT_RX.search(t) else ABSTAIN

@labeling_function()
def lf_rant_never_visit_hearsay(row):
    t = str(row.get('text_clean','') or '')
    # hearsay alone is weak; require a negative orientation or complaint cue
    neg = any(k in t.lower() for k in ["terrible","scam","awful","worst","cheated","fraud","rude","waste","avoid"])
    return RANT if (HEARSAY_RX.search(t) and neg) else ABSTAIN

@labeling_function()
def lf_rant_never_visit_phone_only(row):
    t = str(row.get('text_clean','') or '')
    # phone-only experience + no visit language = likely never inside
    neverish = NEVER_EXPLICIT_RX.search(t) or HEARSAY_RX.search(t) or DRIVEBY_RX.search(t)
    return RANT if (PHONE_ONLY_RX.search(t) and neverish) else ABSTAIN

@labeling_function()
def lf_rant_never_visit_driveby(row):
    t = str(row.get('text_clean','') or '')
    return RANT if DRIVEBY_RX.search(t) else ABSTAIN

@labeling_function()
def lf_rant_never_visit_closed_no_entry(row):
    t = str(row.get('text_clean','') or '')
    # reviewers complaining without actually entering
    return RANT if CLOSED_NO_ENTRY_RX.search(t) else ABSTAIN

@labeling_function()
def lf_rant_never_visit_metadata_boost(row):
    """Boost never-visit when text cues + metadata suggest non-visit."""
    t = str(row.get('text_clean','') or '').lower()
    has_photos = int(row.get('has_photos', 0) or 0)
    wc = len(re.findall(r"\b\w+\b", t))
    cue = NEVER_EXPLICIT_RX.search(t) or DRIVEBY_RX.search(t) or PHONE_ONLY_RX.search(t)
    # no photos + low word count or purely phone/drive-by → likely never entered
    return RANT if (cue and (not has_photos) and wc <= 40) else ABSTAIN

@labeling_function()
def lf_irrelevant_politics(row):
    return IRRELEVANT if POLITICS_RX.search(str(row.get('text_clean','') or '')) else ABSTAIN

@labeling_function()
def lf_irrelevant_employment(row):
    # job ads / hiring chatter inside a review
    return IRRELEVANT if EMPLOYMENT_RX.search(str(row.get('text_clean','') or '')) else ABSTAIN

@labeling_function()
def lf_irrelevant_generic_template(row):
    # ultra-generic one-liners
    t = str(row.get('text_clean','') or '').strip()
    return IRRELEVANT if GENERIC_TEMPLATE_RX.match(t) else ABSTAIN

@labeling_function()
def lf_irrelevant_other_business_compare(row):
    t = str(row.get('text_clean','') or '')
    tl = t.lower()
    wc = len(re.findall(r"\b\w+\b", tl))
    # only consider longer texts with explicit compare markers
    if wc < 12:
        return ABSTAIN
    if re.search(r"\b(vs|compared to|better than|worse than)\b", tl):
        # require at least one Proper-Name-ish token to avoid false matches
        if re.search(r"\b[A-Z][a-z]{2,}\b", t):
            return IRRELEVANT
    return ABSTAIN

@labeling_function()
def lf_irrelevant_too_short(row):
    # very short with no nouns/adjectives often adds no value
    t = str(row.get('text_clean','') or '')
    wc = len(re.findall(r"\b\w+\b", t))
    return IRRELEVANT if wc < 2 else ABSTAIN


@labeling_function()
def lf_promotion_social_handles(row):
    t = str(row.get('text_clean','') or '').lower()
    return PROMOTION if re.search(r"@\w+|\binstagram|facebook|tiktok|follow\s+us\b", t) else ABSTAIN

@labeling_function()
def lf_genuine_domain_hits(row):
    text = str(row.get('text_clean','') or '').lower()
    cat  = str(row.get('category_primary','') or row.get('category_all',''))
    fam  = category_to_family(cat)
    if fam == "generic":
        return ABSTAIN

    toks = re.findall(r"\b[a-z]+\b", text)
    if len(toks) < 6:
        return ABSTAIN

    dom = DOMAIN_TERMS.get(fam, set())
    hits = sum(1 for w in toks if w in dom)
    # Tunable: 3+ clear domain words → likely on-topic genuine
    return GENUINE if hits >= 3 else ABSTAIN

@labeling_function()
def lf_genuine_rating_len(row):
    rating = row.get('rating', None)
    text   = str(row.get('text_clean','') or '')
    wc     = len(re.findall(r"\b\w+\b", text))
    if pd.notna(rating) and rating >= 4 and wc >= 15:
        return GENUINE
    return ABSTAIN

@labeling_function()
def lf_genuine_photos_or_response(row):
    has_photos  = int(row.get('has_photos', 0) or 0)
    photo_count = int(row.get('photo_count', 0) or 0)
    has_resp    = int(row.get('has_response', 0) or 0)
    # Any concrete engagement signal → likely genuine
    if (has_photos and photo_count > 0) or has_resp:
        return GENUINE
    return ABSTAIN

@labeling_function()
def lf_rating_based_genuine(row):
    try:
        r = float(row.get('rating', 0))
    except:
        r = 0
    wc = row.get('word_count', 0)
    if r >= 4 and wc >= 5:
        return GENUINE
    return ABSTAIN


# ----------------------------
# Create a list of all LFs to use later
# ----------------------------
lfs = [
    # --- promotion / ads ---
    lf_contains_url_phone_email,
    lf_contains_promo_words,
    lf_promotion_social_handles,

    # --- irrelevant ---
    lf_offdomain_by_category,
    lf_irrelevant_generic_template,
    lf_irrelevant_other_business_compare,
    lf_irrelevant_politics,
    lf_irrelevant_employment,
    lf_irrelevant_too_short,
    lf_generic_short_review,  # keep if you like this heuristic

    # --- rant: never-visited specific ---
    lf_rant_never_visit_explicit,
    lf_rant_never_visit_hearsay,
    lf_rant_never_visit_phone_only,
    lf_rant_never_visit_driveby,
    lf_rant_never_visit_closed_no_entry,
    lf_rant_never_visit_metadata_boost,

    # --- positive / genuine ---
    lf_high_quality_signal,
    lf_genuine_first_person,
    lf_genuine_descriptive,
    lf_genuine_domain_hits,
    lf_genuine_rating_len,             # NEW
    lf_genuine_photos_or_response,
    lf_rating_based_genuine
]