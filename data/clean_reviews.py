import gzip
import json
import pandas as pd

# ----------------------------
# Step 1: Generator to read JSONL.gz safely
# ----------------------------


def parse_gzip_jsonl(path):
    """Yield one JSON object (as a dict) at a time from a gzip-compressed JSONL file."""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj is not None:
                    yield obj
            except json.JSONDecodeError:
                continue  # skip invalid lines

# ----------------------------
# Step 2: Process each review safely
# ----------------------------


def process_review(review):
    if review is None:
        return None

    # Handle missing pics and resp
    pics = review.get('pics') or []
    resp = review.get('resp') or {'time': None, 'text': ''}

    # Extract first picture URL
    first_pic_url = None
    if pics and isinstance(pics, list) and 'url' in pics[0]:
        first_pic_url = pics[0]['url'][0] if pics[0]['url'] else None

    # Convert timestamps
    review_time = pd.to_datetime(review.get(
        'time'), unit='ms', errors='coerce')
    resp_time = pd.to_datetime(resp.get('time'), unit='ms', errors='coerce')

    # Clean text fields
    text = review.get('text') or ''
    resp_text = resp.get('text') or ''
    name = review.get('name') or ''

    text = text.strip()
    resp_text = resp_text.strip()
    name = name.strip()

    # Rating filter
    rating = review.get('rating')
    if rating is None or not (1 <= rating <= 5):
        return None  # skip invalid ratings

    return {
        'user_id': review.get('user_id'),
        'name': name,
        'review_time': review_time,
        'rating': rating,
        'text': text,
        'first_pic_url': first_pic_url,
        'resp_text': resp_text,
        'resp_time': resp_time,
        'gmap_id': review.get('gmap_id')
    }


# ----------------------------
# Step 3: Stream process and write CSV in chunks
# ----------------------------
# replace with your absolute path
input_file = '/Users/peienlee/Documents/GitHub/Tiktok-TechJam-2025/data/review-Alaska.json.gz'
output_file = '/Users/peienlee/Documents/GitHub/Tiktok-TechJam-2025/data/cleaned_reviews.csv'
chunk_size = 10000  # adjust if needed

rows = []
with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
    header_written = False

    for i, review in enumerate(parse_gzip_jsonl(input_file), start=1):
        if review is None:
            continue  # skip bad lines
        cleaned = process_review(review)
        if cleaned:
            rows.append(cleaned)

        # Write in chunks
        if i % chunk_size == 0:
            df_chunk = pd.DataFrame(rows)
            if not header_written:
                df_chunk.to_csv(f_out, index=False)
                header_written = True
            else:
                df_chunk.to_csv(f_out, index=False, header=False)
            rows = []
            print(f"Processed {i} reviews...")

    # Write any remaining rows
    if rows:
        df_chunk = pd.DataFrame(rows)
        if not header_written:
            df_chunk.to_csv(f_out, index=False)
        else:
            df_chunk.to_csv(f_out, index=False, header=False)

print(f"Cleaned data saved to '{output_file}'")
