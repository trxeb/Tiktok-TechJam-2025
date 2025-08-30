import pandas as pd
import numpy as np
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib 
from labelling.labelling_functions import lfs, ABSTAIN, GENUINE, PROMOTION, IRRELEVANT, RANT

# ----------------------------
# Train Label Model & Classifier
# ----------------------------
def main():
    # 1. Load data
    df = pd.read_csv("googlelocal_reviews_cleaned.csv")
    df["text_clean"] = df["text_clean"].fillna("").astype(str)

    # 2. Apply LFs with progress bar
    print("Applying labeling functions...")
    L_train = []
    for i in tqdm(range(len(df)), desc="Applying LFs"):
        row = df.iloc[i]
        row_labels = [lf(row) for lf in lfs]
        L_train.append(row_labels)
    L_train = np.array(L_train)
    print("Labeling functions applied!")

    # Optional: Analyze LF coverage
    coverage = (L_train != ABSTAIN).any(axis=1).mean()
    print(f"Overall LF coverage: {coverage*100:.2f}%")

    # 3. Train Label Model
    print("Training Snorkel LabelModel...")
    label_model = LabelModel(cardinality=4, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    # 4. Generate weak labels
    df["weak_label"] = label_model.predict(L=L_train, tie_break_policy="abstain")
    print("Weak labels generated!")

    # 5. Load pre-extracted features
    df_features = pd.read_csv("googlelocal_review_features_flagged.csv")

    # 6. Train a classifier
    X = df_features.drop(columns=["text"])
    y = df["weak_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForestClassifier...")
    # Initialize classifier with warm_start=True
    clf = RandomForestClassifier(
        n_estimators=1,  # start with 1 tree
        warm_start=True,
        class_weight={-1:1, 0:5, 1:2, 2:1, 3:2},  # higher weight to Genuine
        random_state=42
)

    # Incrementally add trees with progress bar
    n_trees = 200
    for i in tqdm(range(1, n_trees + 1), desc="Training Random Forest"):
        clf.n_estimators = i
        clf.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save weak labels and model
    df_features["weak_label"] = df["weak_label"]
    df_features.to_csv("reviews_features_with_weak_labels.csv", index=False)
    print("Weak labels and features saved!")

    # 9. Save trained model
    joblib.dump(clf, "random_forest_model.pkl")
    print("Trained Random Forest saved as random_forest_model.pkl")

if __name__ == "__main__":
    main()
