# label_model_training.py
import pandas as pd
import numpy as np
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
import matplotlib.pyplot as plt
import seaborn as sns

# Import your labeling functions
from labelling_functions import lfs, ABSTAIN, GENUINE, PROMOTION, IRRELEVANT, RANT

def main():
    # ----------------------------
    # 1. Load and Prepare Data
    # ----------------------------
    print("Loading data...")
    df = pd.read_csv("googlelocal_reviews_cleaned.csv")
    
    # Clean and prepare the text column
    df["text_clean"] = df["text_clean"].replace("None", np.nan).fillna("").astype(str)
    
    # Filter out empty texts to avoid errors
    original_size = len(df)
    df = df[df["text_clean"].str.strip() != ""].copy()
    print(f"Removed {original_size - len(df)} empty texts. Working with {len(df)} samples.")
    
    # ----------------------------
    # 2. Apply Labeling Functions
    # ----------------------------
    print("Applying labeling functions...")
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)
    
    # ----------------------------
    # 3. Analyze Labeling Functions
    # ----------------------------
    print("\n=== LABELING FUNCTION ANALYSIS ===")
    
    # Get summary statistics
    lf_summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print(lf_summary)
    
    # Calculate overall coverage
    total_coverage = (L_train != ABSTAIN).any(axis=1).mean() * 100
    print(f"\nOverall coverage: {total_coverage:.2f}% of data labeled by at least one LF")
    
    # Plot LF coverage
    plt.figure(figsize=(10, 6))
    coverage = (L_train != ABSTAIN).mean(axis=0) * 100
    sns.barplot(x=[lf.name for lf in lfs], y=coverage)
    plt.title('Labeling Function Coverage')
    plt.ylabel('Percentage of Data Labeled (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('lf_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----------------------------
    # 4. Train the Label Model
    # ----------------------------
    print("\n=== TRAINING LABEL MODEL ===")
    
    # Initialize and train the label model
    label_model = LabelModel(
        cardinality=4,  # 4 classes: GENUINE, PROMOTION, IRRELEVANT, RANT
        verbose=True
    )
    
    label_model.fit(
        L_train=L_train,
        n_epochs=500,
        log_freq=100,
        seed=123
    )
    
    # ----------------------------
    # 5. Generate Predictions
    # ----------------------------
    print("\n=== GENERATING PREDICTIONS ===")
    
    # Get hard labels (most probable class)
    df["weak_label"] = label_model.predict(L=L_train, tie_break_policy="abstain")
    
    # Get probabilistic labels (confidence scores)
    probs = label_model.predict_proba(L=L_train)
    df["prob_genuine"] = probs[:, GENUINE]
    df["prob_promotion"] = probs[:, PROMOTION]
    df["prob_irrelevant"] = probs[:, IRRELEVANT]
    df["prob_rant"] = probs[:, RANT]
    
    # Get the confidence of the predicted label
    df["confidence"] = np.max(probs, axis=1)
    
    # ----------------------------
    # 6. Analyze Results
    # ----------------------------
    print("\n=== RESULTS ANALYSIS ===")
    
    # Distribution of predicted labels
    label_counts = df["weak_label"].value_counts().sort_index()
    label_names = {
        ABSTAIN: "ABSTAIN",
        GENUINE: "GENUINE",
        PROMOTION: "PROMOTION", 
        IRRELEVANT: "IRRELEVANT",
        RANT: "RANT"
    }
    
    print("\nLabel Distribution:")
    for label_num, count in label_counts.items():
        label_name = label_names.get(label_num, f"UNKNOWN_{label_num}")
        percentage = (count / len(df)) * 100
        print(f"{label_name}: {count} samples ({percentage:.2f}%)")
    
    # Confidence statistics
    print(f"\nAverage confidence: {df['confidence'].mean():.3f}")
    print(f"Confidence std: {df['confidence'].std():.3f}")
    
    # Plot label distribution
    plt.figure(figsize=(10, 6))
    label_names_plot = [label_names.get(i, f"UNK_{i}") for i in label_counts.index]
    sns.barplot(x=label_names_plot, y=label_counts.values)
    plt.title('Weak Label Distribution')
    plt.ylabel('Number of Samples')
    plt.xlabel('Label Class')
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----------------------------
    # 7. Save Results
    # ----------------------------
    print("\n=== SAVING RESULTS ===")
    
    # Save the full dataset with weak labels
    output_file = "googlelocal_reviews_weakly_labeled.csv"
    df.to_csv(output_file, index=False)
    
    # Save a sample for manual inspection
    sample_size = min(100, len(df))
    sample_df = df.sample(sample_size, random_state=42)[[
        'text_clean', 'weak_label', 'confidence', 
        'prob_genuine', 'prob_promotion', 'prob_irrelevant', 'prob_rant'
    ]]
    sample_df.to_csv("weak_labels_sample_for_review.csv", index=False)
    
    # Save the label model for future use
    # You can use joblib or pickle to save the model if needed
    
    print(f"✓ Weak labels saved to: {output_file}")
    print(f"✓ Sample for review saved to: weak_labels_sample_for_review.csv")
    print(f"✓ Analysis plots saved as: lf_coverage.png and label_distribution.png")

if __name__ == "__main__":
    main()