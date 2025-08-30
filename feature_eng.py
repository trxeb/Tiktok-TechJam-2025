# tfidf_feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

def main():
    print("🚀 Starting TF-IDF Feature Engineering")
    start_time = time.time()
    
    # ----------------------------
    # 1. Load Data
    # ----------------------------
    print("📂 Loading data...")
    df = pd.read_csv("Data labeling\googlelocal_reviews_weakly_labeled.csv")
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    
    # Remove any empty texts just in case
    df = df[df["text_clean"].str.strip() != ""].copy()
    print(f"📊 Working with {len(df):,} reviews")
    
    # ----------------------------
    # 2. Create TF-IDF Features
    # ----------------------------
    print("🔧 Creating TF-IDF features...")
    
    # Configure TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=2000,          # Keep top 2000 features
        ngram_range=(1, 2),         # Single words and bigrams
        min_df=5,                   # Ignore terms that appear in <5 documents
        max_df=0.8,                 # Ignore terms that appear in >80% of documents
        stop_words='english',       # Remove common English words
        sublinear_tf=True,          # Use 1+log(tf) instead of raw counts
        norm='l2'                   # Normalize vectors
    )
    
    # Fit and transform the text data
    print("   Transforming text to TF-IDF matrix...")
    X_tfidf = tfidf.fit_transform(df['text_clean'])
    
    print(f"   Created TF-IDF matrix with shape: {X_tfidf.shape}")
    print(f"   Vocabulary size: {len(tfidf.get_feature_names_out())}")
    
    # ----------------------------
    # 3. Dimensionality Reduction (Optional but Recommended)
    # ----------------------------
    print("📉 Reducing dimensionality with SVD...")
    
    # Reduce to 100 components (captures most variance)
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    
    print(f"   Reduced to {X_reduced.shape[1]} components")
    print(f"   Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
    
    # ----------------------------
    # 4. Create Features DataFrame
    # ----------------------------
    print("📋 Creating features DataFrame...")
    
    # Create main features
    features_df = pd.DataFrame(
        X_reduced, 
        columns=[f'tfidf_svd_{i}' for i in range(X_reduced.shape[1])]
    )
    
    # Add metadata
    features_df['weak_label'] = df['weak_label'].values
    features_df['confidence'] = df['confidence'].values
    features_df['text_length'] = df['text_clean'].str.len().values
    
    # ----------------------------
    # 5. Save Results
    # ----------------------------
    print("💾 Saving results...")
    
    # Save features
    features_df.to_csv("googlelocal_reviews_tfidf_features.csv", index=False)
    
    # Save vocabulary for reference
    vocab_df = pd.DataFrame({
        'feature_name': tfidf.get_feature_names_out(),
        'feature_index': range(len(tfidf.get_feature_names_out()))
    })
    vocab_df.to_csv("tfidf_vocabulary.csv", index=False)
    
    # Save SVD components for interpretation
    component_df = pd.DataFrame(
        svd.components_[:10],  # First 10 components
        columns=tfidf.get_feature_names_out(),
        index=[f'svd_component_{i}' for i in range(10)]
    )
    component_df.to_csv("svd_top_components.csv")
    
    # ----------------------------
    # 6. Analysis and Visualization
    # ----------------------------
    print("📈 Generating analysis...")
    
    # Label distribution
    plt.figure(figsize=(10, 6))
    label_counts = features_df['weak_label'].value_counts().sort_index()
    label_names = {
        -1: "ABSTAIN",
        0: "GENUINE", 
        1: "PROMOTION",
        2: "IRRELEVANT",
        3: "RANT"
    }
    
    label_names_plot = [label_names.get(i, f"UNK_{i}") for i in label_counts.index]
    sns.barplot(x=label_names_plot, y=label_counts.values)
    plt.title('Label Distribution in Feature Set')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('label_distribution_tfidf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(features_df['confidence'], bins=50)
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----------------------------
    # 7. Final Summary
    # ----------------------------
    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    
    print(f"\n✅ FEATURE ENGINEERING COMPLETE!")
    print(f"⏰ Total time: {total_minutes:.1f} minutes")
    print(f"📊 Features saved: googlelocal_reviews_tfidf_features.csv")
    print(f"📝 Vocabulary saved: tfidf_vocabulary.csv")
    print(f"🎯 Final feature set: {features_df.shape[1]} features")
    print(f"📈 Sample size: {features_df.shape[0]:,} reviews")
    
    # Show label distribution
    print(f"\n📋 Label Distribution:")
    for label_num, count in label_counts.items():
        label_name = label_names.get(label_num, f"UNKNOWN_{label_num}")
        percentage = (count / len(features_df)) * 100
        print(f"   {label_name}: {count:,} samples ({percentage:.1f}%)")

if __name__ == "__main__":
    main()