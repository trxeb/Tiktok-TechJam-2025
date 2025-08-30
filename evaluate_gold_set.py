# evaluate_gold_set.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🧪 Evaluating Model on Gold Set")
    
    # ----------------------------
    # 1. Load Gold Set Data
    # ----------------------------
    print("📂 Loading gold set data...")
    gold_df = pd.read_csv("gold_set_labeled.csv")  # Replace with your gold set path
    gold_df["text_clean"] = gold_df["text_clean"].fillna("").astype(str)
    
    # Remove any empty texts
    gold_df = gold_df[gold_df["text_clean"].str.strip() != ""].copy()
    print(f"📊 Gold set size: {len(gold_df):,} reviews")
    
    # ----------------------------
    # 2. Load Model and Vectorizers
    # ----------------------------
    print("🔧 Loading model and vectorizers...")
    try:
        model = joblib.load('review_classifier_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        svd = joblib.load('svd_transformer.pkl')
        print("✅ All components loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # ----------------------------
    # 3. Preprocess Gold Set (Same as training)
    # ----------------------------
    print("🔄 Preprocessing gold set text...")
    X_gold_tfidf = tfidf.transform(gold_df['text_clean'])
    X_gold_reduced = svd.transform(X_gold_tfidf)
    
    # Create features DataFrame (same structure as training)
    X_gold = pd.DataFrame(
        X_gold_reduced, 
        columns=[f'tfidf_svd_{i}' for i in range(X_gold_reduced.shape[1])]
    )
    
    # Add text length feature
    X_gold['text_length'] = gold_df['text_clean'].str.len().values
    
    # Get true labels
    y_true = gold_df['gold_label']  # Replace with your gold label column name
    
    # ----------------------------
    # 4. Make Predictions
    # ----------------------------
    print("🔮 Making predictions...")
    y_pred = model.predict(X_gold)
    y_pred_proba = model.predict_proba(X_gold)
    
    # ----------------------------
    # 5. Evaluate Performance
    # ----------------------------
    print("📊 Evaluating performance...")
    
    label_names = {
        0: "GENUINE", 
        1: "PROMOTION",
        2: "IRRELEVANT", 
        3: "RANT"
    }
    
    print("\n" + "="*60)
    print("GOLD SET EVALUATION RESULTS")
    print("="*60)
    
    # Classification report
    print("\n📈 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, 
                               target_names=[label_names[i] for i in sorted(np.unique(y_true))],
                               digits=4))
    
    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=[label_names[i] for i in sorted(np.unique(y_true))],
                yticklabels=[label_names[i] for i in sorted(np.unique(y_true))],
                cmap="Blues", cbar=False)
    plt.title('Confusion Matrix - Gold Set Evaluation', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('gold_set_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved: gold_set_confusion_matrix.png")
    
    # ----------------------------
    # 6. Detailed Analysis
    # ----------------------------
    print("\n🔍 DETAILED ANALYSIS:")
    
    # Accuracy by class
    accuracy_by_class = {}
    for class_label in np.unique(y_true):
        class_mask = y_true == class_label
        accuracy = np.mean(y_pred[class_mask] == class_label)
        accuracy_by_class[label_names[class_label]] = accuracy
        print(f"   {label_names[class_label]} accuracy: {accuracy:.3f}")
    
    # Most confident predictions analysis
    max_proba = np.max(y_pred_proba, axis=1)
    confidence_threshold = 0.8
    high_confidence_mask = max_proba > confidence_threshold
    high_confidence_accuracy = np.mean(y_pred[high_confidence_mask] == y_true[high_confidence_mask])
    
    print(f"\n   High confidence (>0.8) predictions: {np.sum(high_confidence_mask):,}/{len(y_true):,}")
    print(f"   High confidence accuracy: {high_confidence_accuracy:.3f}")
    
    # ----------------------------
    # 7. Save Results
    # ----------------------------
    print("\n💾 Saving evaluation results...")
    
    # Create results DataFrame
    results_df = gold_df.copy()
    results_df['predicted_label'] = y_pred
    results_df['prediction_confidence'] = max_proba
    results_df['correct'] = (y_pred == y_true).astype(int)
    
    # Add predicted class names
    results_df['predicted_class'] = results_df['predicted_label'].map(label_names)
    
    # Save detailed results
    results_df.to_csv('gold_set_evaluation_results.csv', index=False)
    print("✅ Detailed results saved: gold_set_evaluation_results.csv")
    
    # ----------------------------
    # 8. Final Summary
    # ----------------------------
    overall_accuracy = np.mean(y_pred == y_true)
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   Overall accuracy: {overall_accuracy:.4f}")
    print(f"   Gold set size: {len(y_true):,} samples")
    print(f"   Model: Random Forest")
    print(f"   Evaluation complete! 🎉")

if __name__ == "__main__":
    main()