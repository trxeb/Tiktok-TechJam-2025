import pandas as pd
import numpy as np
import jsonlines
from tqdm import tqdm
import re
from datetime import datetime
import os

# Load the JSON Lines file
def load_jsonl_file(file_path):
    """
    Load JSON Lines file into a pandas DataFrame
    """
    data = []
    print("Loading JSON Lines file...")
    
    with jsonlines.open(file_path) as reader:
        for obj in tqdm(reader, desc="Reading records"):
            data.append(obj)
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df

# Data cleaning functions
def clean_review_data(df):
    """
    Comprehensive cleaning for review data
    """
    df_clean = df.copy()
    
    print("Starting data cleaning...")
    
    # 1. Convert timestamp to datetime
    if 'time' in df_clean.columns:
        print("Converting timestamps...")
        df_clean['timestamp'] = pd.to_datetime(df_clean['time'], unit='ms')
        df_clean['date'] = df_clean['timestamp'].dt.date
        df_clean['year'] = df_clean['timestamp'].dt.year
        df_clean['month'] = df_clean['timestamp'].dt.month
    
    # 2. Handle missing values
    print("Handling missing values...")
    
    # Fill numeric columns
    if 'rating' in df_clean.columns:
        df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())
    
    # Fill text columns
    text_columns = ['text', 'resp', 'name']
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('None')
    
    # Handle null pics
    if 'pics' in df_clean.columns:
        df_clean['has_photos'] = df_clean['pics'].notnull()
        df_clean['photo_count'] = df_clean['pics'].apply(lambda x: len(x) if x is not None else 0)
    
    # 3. Clean text data
    print("Cleaning text data...")
    
    if 'text' in df_clean.columns:
        df_clean['text_clean'] = df_clean['text'].str.strip()
        df_clean['text_clean'] = df_clean['text_clean'].str.replace(r'\s+', ' ', regex=True)
        df_clean['text_length'] = df_clean['text_clean'].str.len()
        df_clean['word_count'] = df_clean['text_clean'].str.split().str.len()
    
    if 'name' in df_clean.columns:
        df_clean['name_clean'] = df_clean['name'].str.strip().str.title()
    
    if 'resp' in df_clean.columns:
        df_clean['has_response'] = df_clean['resp'].notnull()
        df_clean['response_length'] = df_clean['resp'].str.len().fillna(0)
    
    # 4. Remove duplicates
    print("Removing duplicates...")
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['user_id', 'gmap_id', 'time'], keep='first')
    print(f"Removed {initial_count - len(df_clean)} duplicate records")
    
    # 5. Create useful features
    print("Creating additional features...")
    
    # User activity features
    if 'user_id' in df_clean.columns:
        user_review_counts = df_clean['user_id'].value_counts()
        df_clean['user_review_count'] = df_clean['user_id'].map(user_review_counts)
    
    # Business review counts
    if 'gmap_id' in df_clean.columns:
        business_review_counts = df_clean['gmap_id'].value_counts()
        df_clean['business_review_count'] = df_clean['gmap_id'].map(business_review_counts)
    
    # 6. Filter out outliers
    print("Filtering outliers...")
    
    if 'text_length' in df_clean.columns:
        # Remove extremely short or long reviews
        q1 = df_clean['text_length'].quantile(0.01)
        q3 = df_clean['text_length'].quantile(0.99)
        df_clean = df_clean[(df_clean['text_length'] >= q1) & (df_clean['text_length'] <= q3)]
    
    if 'rating' in df_clean.columns:
        # Ensure ratings are within valid range (1-5)
        df_clean = df_clean[df_clean['rating'].between(1, 5)]
    
    print("Data cleaning complete!")
    return df_clean

# Analysis functions
def analyze_data(df):
    """
    Generate basic analysis of the cleaned data
    """
    print("\n" + "="*50)
    print("DATA ANALYSIS REPORT")
    print("="*50)
    
    print(f"Total reviews: {len(df):,}")
    print(f"Total users: {df['user_id'].nunique():,}")
    print(f"Total businesses: {df['gmap_id'].nunique():,}")
    
    if 'rating' in df.columns:
        print(f"\nRating distribution:")
        print(df['rating'].value_counts().sort_index())
        print(f"Average rating: {df['rating'].mean():.2f}")
    
    if 'text_length' in df.columns:
        print(f"\nReview length statistics:")
        print(f"Average characters: {df['text_length'].mean():.1f}")
        print(f"Average words: {df['word_count'].mean():.1f}")
    
    if 'timestamp' in df.columns:
        print(f"\nTime range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    if 'has_photos' in df.columns:
        print(f"\nReviews with photos: {df['has_photos'].sum():,} ({df['has_photos'].mean()*100:.1f}%)")
    
    if 'has_response' in df.columns:
        print(f"Reviews with responses: {df['has_response'].sum():,} ({df['has_response'].mean()*100:.1f}%)")

# Main execution
def main():
    # Update this path to your actual file location
    file_path = "review-Wyoming_10.json"  # or whatever your file is called
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please update the file_path variable to point to your downloaded file")
        return
    
    # Load the data
    df = load_jsonl_file(file_path)
    
    # Show initial data info
    print("\nInitial data info:")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_review_data(df)
    
    # Analyze cleaned data
    analyze_data(cleaned_df)
    
    # Save cleaned data
    output_file = "googlelocal_reviews_cleaned.csv"
    cleaned_df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")
    
    # Save a sample for quick inspection
    sample_file = "googlelocal_reviews_sample.csv"
    cleaned_df.head(1000).to_csv(sample_file, index=False)
    print(f"Sample data saved to: {sample_file}")

if __name__ == "__main__":
    main()

