import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_with_gold_set():
    print("🤖 Evaluating Model with Gold Set")
    
    # ----------------------------
    # Load the trained model
    # ----------------------------
    print("📂 Loading trained model...")
    try:
        model = joblib.load('review_classifier_model.pkl')
        print("✅ Model loaded successfully")
    except:
        print("❌ Model file not found. Please train the model first.")
        return
    
    # ----------------------------
    # Load the gold set
    # ----------------------------
    print("📂 Loading gold set...")
    gold_df = pd.read_csv("gold_set_done_trixie.csv")
    
    # Check if we have the required columns
    if 'gold_label' not in gold_df.columns or 'text_clean' not in gold_df.columns:
        print("❌ Gold set missing required columns ('gold_label' or 'text_clean')")
        return
    
    print(f"📊 Gold set contains {gold_df.shape[0]:,} samples")
    print("Gold set class distribution:")
    print(gold_df['gold_label'].value_counts())
    
    # ----------------------------
    # Load the TF-IDF vectorizer
    # ----------------------------
    print("📂 Loading TF-IDF vectorizer...")
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("✅ Vectorizer loaded successfully")
    except:
        print("❌ Vectorizer not found. Please run training first to create it.")
        return
    
    # ----------------------------
    # Prepare the gold set features
    # ----------------------------
    print("🔧 Preparing gold set features...")
    
    # Transform the text using the vectorizer
    try:
        X_gold = vectorizer.transform(gold_df['text_clean'])
        print(f"✅ Transformed text into {X_gold.shape[1]} features")
    except Exception as e:
        print(f"❌ Error transforming text: {e}")
        return
    
    # ----------------------------
    # Make predictions
    # ----------------------------
    print("🔮 Making predictions on gold set...")
    y_pred = model.predict(X_gold)
    y_true = gold_df['gold_label'].values
    
    # ----------------------------
    # Evaluate performance
    # ----------------------------
    print("\n" + "="*60)
    print("GOLD SET EVALUATION RESULTS")
    print("="*60)
    
    label_names = {
        0: "GENUINE", 
        1: "PROMOTION", 
        2: "IRRELEVANT", 
        3: "RANT"
    }
    
    # Filter out classes not present in the gold set
    present_classes = np.unique(y_true)
    target_names = [label_names[i] for i in present_classes if i in label_names]
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.3f}")
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=list(present_classes))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=target_names,
                yticklabels=target_names,
                cmap="Blues")
    plt.title('Confusion Matrix - Gold Set Evaluation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('gold_set_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved as: gold_set_confusion_matrix.png")
    
    # ----------------------------
    # Error analysis
    # ----------------------------
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    # Create a DataFrame for error analysis
    results_df = gold_df.copy()
    results_df['predicted_label'] = y_pred
    results_df['is_correct'] = (y_true == y_pred)
    
    # Incorrect predictions
    incorrect_df = results_df[~results_df['is_correct']]
    print(f"Number of incorrect predictions: {len(incorrect_df)}")
    
    if len(incorrect_df) > 0:
        print("\nSample of incorrect predictions:")
        for i, row in incorrect_df.head(5).iterrows():
            print(f"\nText: {row['text_clean'][:100]}...")
            print(f"True: {label_names.get(row['gold_label'], 'UNKNOWN')}, Predicted: {label_names.get(row['predicted_label'], 'UNKNOWN')}")
    
    # Save detailed results
    results_df.to_csv('gold_set_predictions.csv', index=False)
    print("✅ Detailed results saved as: gold_set_predictions.csv")
    
    # ----------------------------
    # Class-wise performance
    # ----------------------------
    print("\n" + "="*60)
    print("CLASS-WISE PERFORMANCE")
    print("="*60)
    
    for cls in present_classes:
        if cls in label_names:
            cls_mask = y_true == cls
            if np.sum(cls_mask) > 0:
                cls_accuracy = accuracy_score(y_true[cls_mask], y_pred[cls_mask])
                print(f"{label_names[cls]}: {cls_accuracy:.3f} ({np.sum(cls_mask)} samples)")
            else:
                print(f"{label_names[cls]}: No samples")

if __name__ == "__main__":
    evaluate_with_gold_set()