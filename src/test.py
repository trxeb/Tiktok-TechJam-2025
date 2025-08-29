import pandas as pd

# Load your feature CSV
df = pd.read_csv("googlelocal_review_features_flagged.csv")

# Show all column names (so you can confirm embeddings are there)
print("Total columns:", len(df.columns))
print(df.columns.tolist())

# Display the first 5 rows with ALL features (including embeddings)
pd.set_option("display.max_columns", None)   # don't truncate columns
pd.set_option("display.width", 2000)         # wide display
print(df.head(5))

