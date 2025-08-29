import pandas as pd

# Load the full features CSV
features_df = pd.read_csv("googlelocal_review_features_flagged.csv")

# Filter out embedding columns (those containing '_emb_')
non_embedding_cols = [col for col in features_df.columns if "_emb_" not in col]

# Select only the first 5 reviews
first_5_reviews = features_df[non_embedding_cols].head(5)

# Save to a new CSV
first_5_reviews.to_csv("googlelocal_review_features_first5.csv", index=False)

print("Saved first 5 reviews without embeddings to googlelocal_review_features_first5.csv")

