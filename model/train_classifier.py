import os
import pandas as pd
import numpy as np
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import from your labeling functions
from labelling.labelling_functions import lfs, ABSTAIN, GENUINE, PROMOTION, IRRELEVANT, RANT

def load_and_prepare_data(reviews_file, features_file):
    """Load review data and features, ensuring alignment"""
    print("Loading data...")
    
    # Load reviews
    df_reviews = pd.read_csv(reviews_file)
    df_reviews["text_clean"] = df_reviews["text_clean"].fillna("").astype(str)
    
    # Load features
    df_features = pd.read_csv(features_file)
    
    # Ensure same number of samples
    min_samples = min(len(df_reviews), len(df_features))
    df_reviews = df_reviews.iloc[:min_samples]
    df_features = df_features.iloc[:min_samples]
    
    print(f"Loaded {min_samples} samples")
    return df_reviews, df_features

def apply_labeling_functions(df):
    """Apply Snorkel labeling functions with progress tracking"""
    print("Applying labeling functions...")
    
    # Use PandasLFApplier for better performance
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)
    
    # Analyze LF statistics
    coverage = (L_train != ABSTAIN).any(axis=1).mean()
    conflicts = ((L_train != ABSTAIN).sum(axis=1) > 1).mean()
    
    print(f"Overall LF coverage: {coverage*100:.2f}%")
    print(f"Conflicting labels: {conflicts*100:.2f}%")
    
    # Show label distribution for each LF
    print("\nLabeling Function Statistics:")
    for i, lf in enumerate(lfs):
        lf_labels = L_train[:, i]
        non_abstain = (lf_labels != ABSTAIN).sum()
        if non_abstain > 0:
            unique_labels, counts = np.unique(lf_labels[lf_labels != ABSTAIN], return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            print(f"{lf.name}: {non_abstain} labels, distribution: {label_dist}")
    
    return L_train

def train_label_model(L_train):
    """Train Snorkel LabelModel with optimized parameters"""
    print("Training Snorkel LabelModel...")
    
    label_model = LabelModel(cardinality=4, verbose=True)
    label_model.fit(
        L_train=L_train, 
        n_epochs=500, 
        log_freq=100, 
        seed=123, 
        lr=0.01,
        l2=0.0,  # L2 regularization
        optimizer='adam'
    )
    
    # Generate weak labels
    weak_labels = label_model.predict(L=L_train, tie_break_policy="abstain")
    probs = label_model.predict_proba(L=L_train)
    
    # Analyze weak label distribution
    unique_labels, counts = np.unique(weak_labels, return_counts=True)
    label_names = {ABSTAIN: 'ABSTAIN', GENUINE: 'GENUINE', PROMOTION: 'PROMOTION', 
                   IRRELEVANT: 'IRRELEVANT', RANT: 'RANT'}
    
    print("\nWeak Label Distribution:")
    for label, count in zip(unique_labels, counts):
        pct = (count / len(weak_labels)) * 100
        print(f"{label_names.get(label, label)}: {count} ({pct:.1f}%)")
    
    return label_model, weak_labels, probs

def prepare_features(df_features, weak_labels, feature_selection=True, scale_features=True):
    """Prepare features for training"""
    print("Preparing features...")
    
    # Remove text columns and non-numeric columns
    feature_columns = df_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target-like columns
    exclude_cols = ['weak_label', 'rating'] if 'rating' not in feature_columns else ['weak_label']
    feature_columns = [col for col in feature_columns if col not in exclude_cols]
    
    X = df_features[feature_columns].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"Using {len(feature_columns)} features")
    
    # Feature selection
    if feature_selection and len(feature_columns) > 100:
        # Remove samples with ABSTAIN labels for feature selection
        non_abstain_mask = weak_labels != ABSTAIN
        if non_abstain_mask.sum() > 50:  # Need enough samples
            selector = SelectKBest(score_func=f_classif, k=min(100, len(feature_columns)))
            X_temp = X[non_abstain_mask]
            y_temp = weak_labels[non_abstain_mask]
            
            selector.fit(X_temp, y_temp)
            selected_features = X.columns[selector.get_support()].tolist()
            X = X[selected_features]
            print(f"Selected top {len(selected_features)} features")
    
    # Feature scaling
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        print("Features scaled")
    
    return X, scaler

def handle_class_imbalance(X, y, method='smote'):
    """Handle class imbalance in the dataset"""
    # Remove ABSTAIN labels for training
    non_abstain_mask = y != ABSTAIN
    X_filtered = X[non_abstain_mask]
    y_filtered = y[non_abstain_mask]
    
    if len(np.unique(y_filtered)) < 2:
        print("Warning: Insufficient class diversity for imbalance handling")
        return X_filtered, y_filtered
    
    print("Handling class imbalance...")
    print("Original class distribution:")
    unique, counts = np.unique(y_filtered, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")
    
    if method == 'smote':
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(counts)-1))
            X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)
        except:
            print("SMOTE failed, using original data")
            X_resampled, y_resampled = X_filtered, y_filtered
    elif method == 'smoteenn':
        try:
            smoteenn = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smoteenn.fit_resample(X_filtered, y_filtered)
        except:
            print("SMOTEENN failed, using original data")
            X_resampled, y_resampled = X_filtered, y_filtered
    else:
        X_resampled, y_resampled = X_filtered, y_filtered
    
    print("Resampled class distribution:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")
    
    return X_resampled, y_resampled

def train_mlp_classifier(X, y, optimize_hyperparams=True):
    """Train MLP classifier with optional hyperparameter optimization"""
    print("Training MLP classifier...")
    
    # Encode labels to ensure they start from 0
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    if optimize_hyperparams and len(X_train) > 100:
        print("Optimizing MLP hyperparameters...")
        
        # Determine appropriate hidden layer sizes based on feature count
        n_features = X_train.shape[1]
        
        param_grid = {
            'hidden_layer_sizes': [
                (100,),
                (200,),
                (100, 50),
                (200, 100),
                (300, 150, 50),
                (min(500, n_features), min(250, n_features//2)),
            ],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500]
        }
        
        # Reduce parameter space for large datasets
        if len(X_train) > 1000:
            param_grid = {
                'hidden_layer_sizes': [(200,), (200, 100), (300, 150)],
                'activation': ['relu'],
                'solver': ['adam'],
                'alpha': [0.0001, 0.001],
                'learning_rate': ['adaptive'],
                'max_iter': [500]
            }
        
        mlp = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
        
        grid_search = GridSearchCV(
            mlp, param_grid, cv=3, scoring='f1_weighted', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_mlp = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
    else:
        # Use optimized default parameters
        n_features = X_train.shape[1]
        hidden_size = min(200, max(50, n_features // 2))
        
        best_mlp = MLPClassifier(
            hidden_layer_sizes=(hidden_size, hidden_size // 2),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        
        print(f"Using default MLP with hidden layers: ({hidden_size}, {hidden_size // 2})")
        best_mlp.fit(X_train, y_train)
    
    return best_mlp, label_encoder, X_train, X_test, y_train, y_test

def evaluate_mlp_model(mlp, label_encoder, X_test, y_test, feature_names=None):
    """Comprehensive MLP model evaluation"""
    print("Evaluating MLP model...")
    
    # Predictions
    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)
    
    # Decode labels back to original form
    y_test_orig = label_encoder.inverse_transform(y_test)
    y_pred_orig = label_encoder.inverse_transform(y_pred)
    
    # Classification report
    print("\nClassification Report:")
    label_names = {GENUINE: 'GENUINE', PROMOTION: 'PROMOTION', 
                   IRRELEVANT: 'IRRELEVANT', RANT: 'RANT'}
    target_names = [label_names.get(label, f'Class_{label}') for label in sorted(label_encoder.classes_)]
    print(classification_report(y_test_orig, y_pred_orig, target_names=target_names))
    
    # F1 scores
    f1_weighted = f1_score(y_test_orig, y_pred_orig, average='weighted')
    f1_macro = f1_score(y_test_orig, y_pred_orig, average='macro')
    print(f"\nWeighted F1: {f1_weighted:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    
    # MLP-specific metrics
    print(f"\nMLP Training Information:")
    print(f"Number of iterations: {mlp.n_iter_}")
    print(f"Training loss: {mlp.loss_:.4f}")
    if hasattr(mlp, 'validation_scores_'):
        print(f"Best validation score: {max(mlp.validation_scores_):.4f}")
    
    # Training curve visualization
    if hasattr(mlp, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(mlp.loss_curve_)
        plt.title('Training Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if hasattr(mlp, 'validation_scores_'):
            plt.subplot(1, 2, 2)
            plt.plot(mlp.validation_scores_)
            plt.title('Validation Score Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Score')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('mlp_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Feature importance approximation using permutation importance
    if feature_names is not None and len(feature_names) <= 50:  # Only for manageable number of features
        from sklearn.inspection import permutation_importance
        print("Computing feature importance (this may take a moment)...")
        
        perm_importance = permutation_importance(
            mlp, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Save feature importance
        feature_importance.to_csv('mlp_feature_importance.csv', index=False)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_orig, y_pred_orig)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('MLP Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return f1_weighted, f1_macro

def compare_with_existing_mlp_model(new_mlp, label_encoder, X_test, y_test, new_f1):
    """Compare with existing MLP model and decide whether to replace"""
    model_path = "mlp_model.pkl"
    encoder_path = "label_encoder.pkl"
    scaler_path = "feature_scaler.pkl"
    
    if os.path.exists(model_path):
        print("Comparing with existing MLP model...")
        try:
            old_mlp = joblib.load(model_path)
            old_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else label_encoder
            
            # Ensure feature compatibility
            if hasattr(old_mlp, 'n_features_in_') and old_mlp.n_features_in_ == X_test.shape[1]:
                y_pred_old = old_mlp.predict(X_test)
                y_pred_old_orig = old_encoder.inverse_transform(y_pred_old)
                y_test_orig = label_encoder.inverse_transform(y_test)
                
                old_f1 = f1_score(y_test_orig, y_pred_old_orig, average="weighted")
                
                print(f"Old MLP Model Weighted F1: {old_f1:.4f}")
                print(f"New MLP Model Weighted F1: {new_f1:.4f}")
                
                if new_f1 > old_f1 + 0.01:  # Require meaningful improvement
                    joblib.dump(new_mlp, model_path)
                    joblib.dump(label_encoder, encoder_path)
                    print("New MLP model is significantly better. Replaced old model.")
                    return True
                else:
                    print("New MLP model did not significantly outperform. Keeping old model.")
                    return False
            else:
                print("Feature mismatch with old MLP model. Saving new model.")
                joblib.dump(new_mlp, model_path)
                joblib.dump(label_encoder, encoder_path)
                return True
        except Exception as e:
            print(f"Error loading old MLP model: {e}. Saving new model.")
            joblib.dump(new_mlp, model_path)
            joblib.dump(label_encoder, encoder_path)
            return True
    else:
        joblib.dump(new_mlp, model_path)
        joblib.dump(label_encoder, encoder_path)
        print("No existing MLP model found. Saved new model.")
        return True

def save_results(df_features, weak_labels, feature_scaler=None):
    """Save results and model artifacts"""
    print("Saving results...")
    
    # Save weak labels with features
    df_results = df_features.copy()
    df_results["weak_label"] = weak_labels
    df_results.to_csv("reviews_features_with_weak_labels.csv", index=False)
    
    # Save scaler if used
    if feature_scaler is not None:
        joblib.dump(feature_scaler, "feature_scaler.pkl")
    
    print("Results saved!")

def main():
    """Main MLP training pipeline"""
    print("Starting enhanced review classification training with MLP...")
    
    # Configuration
    REVIEWS_FILE = "googlelocal_reviews_cleaned.csv"
    FEATURES_FILE = "googlelocal_review_features_comprehensive.csv"  # Updated filename
    
    # Check if files exist
    if not os.path.exists(FEATURES_FILE):
        print(f"Features file {FEATURES_FILE} not found. Please run feature extraction first.")
        # Try alternative filename
        alt_features_file = "googlelocal_review_features_flagged.csv"
        if os.path.exists(alt_features_file):
            FEATURES_FILE = alt_features_file
            print(f"Using alternative features file: {FEATURES_FILE}")
        else:
            print("No features file found. Exiting.")
            return
    
    try:
        # 1. Load and prepare data
        df_reviews, df_features = load_and_prepare_data(REVIEWS_FILE, FEATURES_FILE)
        
        # 2. Apply labeling functions
        L_train = apply_labeling_functions(df_reviews)
        
        # 3. Train label model
        label_model, weak_labels, probs = train_label_model(L_train)
        
        # 4. Prepare features
        X, scaler = prepare_features(df_features, weak_labels)
        
        # 5. Handle class imbalance
        X_balanced, y_balanced = handle_class_imbalance(X, weak_labels, method='smote')
        
        # Skip training if insufficient data
        if len(X_balanced) < 50:  # MLP needs more data than RF
            print("Insufficient training data for MLP after preprocessing. Skipping classifier training.")
            save_results(df_features, weak_labels, scaler)
            return
        
        # 6. Train MLP classifier
        mlp, label_encoder, X_train, X_test, y_train, y_test = train_mlp_classifier(
            X_balanced, y_balanced, optimize_hyperparams=True
        )
        
        # 7. Evaluate MLP model
        f1_weighted, f1_macro = evaluate_mlp_model(mlp, label_encoder, X_test, y_test, X.columns.tolist())
        
        # 8. Compare with existing MLP model
        model_updated = compare_with_existing_mlp_model(mlp, label_encoder, X_test, y_test, f1_weighted)
        
        # 9. Save results
        save_results(df_features, weak_labels, scaler)
        
        print("\n" + "="*50)
        print("MLP TRAINING COMPLETE!")
        print(f"Final Weighted F1 Score: {f1_weighted:.4f}")
        print(f"Final Macro F1 Score: {f1_macro:.4f}")
        print(f"MLP Model Updated: {model_updated}")
        print("="*50)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()