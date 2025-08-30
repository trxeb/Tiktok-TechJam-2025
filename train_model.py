import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
from collections import Counter

def main():
    print("🤖 Training Classification Model")
    
    # ----------------------------
    # Load features
    # ----------------------------
    print("📂 Loading features...")
    features_df = pd.read_csv("Feature Eng/googlelocal_reviews_tfidf_features.csv")
    
    # Prepare data
    X = features_df.drop(['weak_label', 'confidence'], axis=1)
    y = features_df['weak_label']
    
    # Remove abstain votes
    keep_mask = y != -1
    X = X[keep_mask]
    y = y[keep_mask]
    
    print(f"📊 Training on {X.shape[0]:,} samples with {X.shape[1]} features")
    print("Original class distribution:", Counter(y))
    
    # ----------------------------
    # Split data
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ----------------------------
    # SMOTE oversampling
    # ----------------------------
    print("🔄 Applying SMOTE oversampling...")
    sm = SMOTE(random_state=42, k_neighbors=1)  # k_neighbors=1 for tiny classes
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print("Resampled class distribution:", Counter(y_res))
    
    # ----------------------------
    # Compute class weights
    # ----------------------------
    classes = np.unique(y_res)
    weights = compute_class_weight('balanced', classes=classes, y=y_res)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
    print("Class weights:", class_weight_dict)
    
    # ----------------------------
    # Train Random Forest
    # ----------------------------
    print("🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,        # more trees for stability
        max_depth=25,           # slightly deeper tree
        min_samples_split=5,
        class_weight=class_weight_dict,  # <---- weight applied
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_res, y_res)
    
    # ----------------------------
    # Evaluate Model
    # ----------------------------
    print("📊 Evaluating model...")
    y_pred = rf_model.predict(X_test)
    
    label_names = {
        0: "GENUINE", 
        1: "PROMOTION",
        2: "IRRELEVANT", 
        3: "RANT"
    }
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, 
                                target_names=[label_names[i] for i in sorted(y.unique())]))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=[label_names[i] for i in sorted(y.unique())],
                yticklabels=[label_names[i] for i in sorted(y.unique())],
                cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----------------------------
    # Feature Importance
    # ----------------------------
    print("🔍 Analyzing feature importance...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # ----------------------------
    # Save Model
    # ----------------------------
    print("💾 Saving model...")
    joblib.dump(rf_model, 'review_classifier_model.pkl')
    print("✅ Model saved as: review_classifier_model.pkl")
    
    # ----------------------------
    # Cross-Validation Scores
    # ----------------------------
    print("📈 Running cross-validation...")
    cv_scores = cross_val_score(rf_model, X_res, y_res, cv=3, n_jobs=-1)
    print(f"   Cross-validation scores: {cv_scores}")
    print(f"   Mean CV accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

if __name__ == "__main__":
    main()