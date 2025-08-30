import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Import the SAME feature extraction pipeline from your feature extraction script
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_engineer import build_comprehensive_features

# Constants from your labeling functions
ABSTAIN = -1
GENUINE = 0
PROMOTION = 1
IRRELEVANT = 2
RANT = 3

def load_gold_set_and_extract_features(gold_set_path):
    """
    Load gold set and extract the SAME comprehensive features used in training
    """
    print("Loading gold set...")
    gold_df = pd.read_csv(gold_set_path)
    
    print("Available columns:", gold_df.columns.tolist())
    print(f"Gold set shape: {gold_df.shape}")
    print(f"Gold label distribution:\n{gold_df['gold_label'].value_counts()}")
    
    # Check if we have text_clean column
    if 'text_clean' not in gold_df.columns:
        raise ValueError("Gold set must have 'text_clean' column to match training features")
    
    # Save the gold set in the format expected by feature extraction
    temp_file = "temp_gold_for_features.csv"
    gold_df.to_csv(temp_file, index=False)
    
    try:
        # Use the SAME comprehensive feature extraction as training
        print("Extracting comprehensive features (this may take a moment)...")
        features_df = build_comprehensive_features(temp_file, include_embeddings=True)
        
        # Add back the gold labels
        features_df['gold_label'] = gold_df['gold_label'].values
        
        print(f"Feature extraction complete! Generated {len(features_df.columns)} features")
        
        # Clean up temp file
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return features_df
        
    except Exception as e:
        print(f"Comprehensive feature extraction failed: {e}")
        print("This might be due to missing dependencies or model files.")
        
        # Clean up temp file
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Fallback to basic features (but warn about mismatch)
        print("WARNING: Falling back to basic features - results may not be accurate!")
        return create_basic_features(gold_df)

def create_basic_features(df):
    """Create basic features (FALLBACK ONLY - will cause feature mismatch)"""
    import re
    
    print("WARNING: Using basic features only - this will cause feature mismatch!")
    print("For accurate evaluation, ensure comprehensive feature extraction works.")
    
    # Text characteristics
    df['char_count'] = df['text_clean'].str.len()
    df['word_count'] = df['text_clean'].apply(lambda x: len(re.findall(r'\b\w+\b', str(x))))
    df['sentence_count'] = df['text_clean'].apply(lambda x: len(re.split(r'[.!?]+', str(x))))
    
    # Basic flags
    df['has_url'] = df['text_clean'].str.contains(r'http\S+|www\S+', case=False, na=False).astype(int)
    df['has_email'] = df['text_clean'].str.contains(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', case=False, na=False).astype(int)
    df['has_phone'] = df['text_clean'].str.contains(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}', case=False, na=False).astype(int)
    df['has_first_person'] = df['text_clean'].str.contains(r'\b(i|we|my|our)\b', case=False, na=False).astype(int)
    df['has_past_tense'] = df['text_clean'].str.contains(r'\b(was|were|went|visited|ate|had|ordered)\b', case=False, na=False).astype(int)
    
    # Fill any missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def prepare_gold_features(gold_df):
    """Prepare features to match training pipeline exactly"""
    print("Preparing features for evaluation...")
    
    # Load the scaler to get the expected feature names
    try:
        scaler = joblib.load("feature_scaler.pkl")
        expected_features = scaler.feature_names_in_
        print(f"Scaler expects {len(expected_features)} features")
        
        # Get only the features that the scaler expects
        available_features = gold_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target columns
        exclude_cols = ['gold_label', 'weak_label'] 
        available_features = [col for col in available_features if col not in exclude_cols]
        
        # Create feature matrix with expected features
        X = pd.DataFrame(index=gold_df.index)
        
        # Add features that exist
        for feature in expected_features:
            if feature in available_features:
                X[feature] = gold_df[feature]
            else:
                # Add missing features as zeros
                X[feature] = 0.0
                print(f"Warning: Missing feature '{feature}' - filling with zeros")
        
        # Handle missing and infinite values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Apply scaling
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        print("Applied saved feature scaling")
        return X_scaled, expected_features
        
    except FileNotFoundError:
        print("No saved scaler found, preparing features manually...")
        
        # Get only numeric columns (same as training)
        feature_columns = gold_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target-like columns
        exclude_cols = ['gold_label', 'weak_label']
        feature_columns = [col for col in feature_columns if col not in exclude_cols]
        
        X = gold_df[feature_columns].copy()
        
        # Handle missing and infinite values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Using {len(feature_columns)} features without scaling")
        return X, feature_columns

def evaluate_mlp_on_gold_set():
    """Complete evaluation pipeline with consistent feature extraction"""
    
    # 1. Load saved model components
    try:
        print("Loading saved model components...")
        mlp_model = joblib.load('mlp_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        print("✓ Model components loaded successfully")
        
        # Check what features the model expects
        if hasattr(mlp_model, 'n_features_in_'):
            print(f"Model expects {mlp_model.n_features_in_} features")
            
    except FileNotFoundError as e:
        print(f"Error: Could not load model components: {e}")
        print("Make sure you have trained and saved the MLP model first")
        return
    
    # 2. Load and prepare gold set with SAME feature extraction as training
    gold_df = load_gold_set_and_extract_features('gold_set_done_yingjie.csv')
    
    # 3. Prepare features using the exact same pipeline as training
    X_gold, feature_columns = prepare_gold_features(gold_df)
    y_true = gold_df['gold_label'].values
    
    # 4. Verify feature compatibility
    if hasattr(mlp_model, 'n_features_in_'):
        if X_gold.shape[1] != mlp_model.n_features_in_:
            print(f"FEATURE MISMATCH!")
            print(f"Model expects {mlp_model.n_features_in_} features")
            print(f"Gold set has {X_gold.shape[1]} features")
            print("This suggests the feature extraction pipeline is different from training.")
            
            if X_gold.shape[1] < mlp_model.n_features_in_:
                print("Too few features - evaluation may not be accurate")
                return None
        else:
            print("✓ Feature dimensions match perfectly!")
    
    # 5. Make predictions
    print("Making predictions...")
    y_pred_encoded = mlp_model.predict(X_gold)
    y_pred_proba = mlp_model.predict_proba(X_gold)
    
    # 6. Handle label encoding
    try:
        y_true_encoded = label_encoder.transform(y_true)
        use_encoded = True
    except ValueError:
        print("Using gold labels as-is (assuming they match model encoding)")
        y_true_encoded = y_true
        use_encoded = False
    
    # Decode predictions back to original labels
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)
    
    # 7. Calculate metrics
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    f1_weighted = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
    f1_macro = f1_score(y_true_encoded, y_pred_encoded, average='macro')
    
    print("\n" + "="*50)
    print("MLP MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    
    # 8. Detailed classification report
    label_names = {GENUINE: 'GENUINE', PROMOTION: 'PROMOTION', 
                   IRRELEVANT: 'IRRELEVANT', RANT: 'RANT', ABSTAIN: 'ABSTAIN'}
    
    # Get all unique classes present in both predictions and true labels
    all_classes = np.unique(np.concatenate([y_true_encoded, y_pred_encoded]))
    unique_classes = sorted(all_classes)
    target_names = [label_names.get(cls, f'Class_{cls}') for cls in unique_classes]
    
    print(f"\nFound {len(unique_classes)} classes: {unique_classes}")
    print(f"Target names: {target_names}")
    
    print("\nDetailed Classification Report:")
    try:
        print(classification_report(y_true_encoded, y_pred_encoded, 
                                  labels=unique_classes, target_names=target_names))
    except Exception as e:
        print(f"Classification report failed: {e}")
        print("Using basic classification report:")
        print(classification_report(y_true_encoded, y_pred_encoded))
    
    # 9. Confusion Matrix
    cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=unique_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('MLP Model Confusion Matrix on Gold Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('mlp_gold_set_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 10. Save results
    results_df = gold_df.copy()
    results_df['predicted_label'] = y_pred_original
    results_df['prediction_correct'] = (y_true_encoded == y_pred_encoded)
    results_df['prediction_confidence'] = np.max(y_pred_proba, axis=1)
    
    # Add probability columns - only for classes the model actually predicts
    model_classes = label_encoder.classes_
    print(f"Model was trained on {len(model_classes)} classes: {model_classes}")
    print(f"Probability matrix shape: {y_pred_proba.shape}")
    
    for i in range(y_pred_proba.shape[1]):  # Use actual number of probability columns
        if i < len(model_classes):
            class_label = model_classes[i]
            class_name = label_names.get(class_label, f'Class_{class_label}')
            results_df[f'prob_{class_name}'] = y_pred_proba[:, i]
        else:
            results_df[f'prob_unknown_{i}'] = y_pred_proba[:, i]
    
    results_df.to_csv('mlp_gold_set_evaluation_results.csv', index=False)
    print(f"\nDetailed results saved to: mlp_gold_set_evaluation_results.csv")
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }

# Run the evaluation
if __name__ == "__main__":
    print("Starting MLP model evaluation on gold set...")
    results = evaluate_mlp_on_gold_set()
    if results:
        print("\nEvaluation complete!")
    else:
        print("\nEvaluation failed due to feature mismatch.")
        print("Please ensure the same feature extraction is used for both training and evaluation.")