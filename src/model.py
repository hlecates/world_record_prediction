import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc

from sklearn.impute import SimpleImputer

def prepare_data(df: pd.DataFrame, target_record: str = 'world'):
    """Prepare features and target for modeling with missing value handling."""
    # Define features to use
    feature_cols = [
        'fastest_seed_pct_off_world',
        'fastest_seed_pct_off_american',
        'num_entries',
        'top_seed_is_record_holder',
        'is_sprint', 'is_distance',
        'stroke_encoded',
        'season_peak',
        'num_sub_1pct_off_record',
        'top_3_spread'
    ]
    
    # Add positional features
    for pos in [1, 3, 5, 8]:
        feature_cols.extend([
            f'pos_{pos}_pct_off_world',
            f'pos_{pos}_pct_off_american',
            f'pos_{pos}_pct_off_us_open'
        ])
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[f'{target_record}_record_broken']
    
    # Print missing value information
    print("\nMissing values before imputation:")
    print(X.isnull().sum()[X.isnull().sum() > 0])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print("\nMissing values after imputation:")
    print(X_imputed.isnull().sum()[X_imputed.isnull().sum() > 0])
    
    return X_imputed, y

def load_features():
    """Load engineered features dataset."""
    base_dir = Path(__file__).parent.parent
    features_path = base_dir / "data" / "features" / "record_prediction_features.csv"
    return pd.read_csv(features_path)

def prepare_data(df: pd.DataFrame, target_record: str = 'world'):
    """Prepare features and target for modeling."""
    # Define features to use
    feature_cols = [
        'fastest_seed_pct_off_world',
        'fastest_seed_pct_off_american',
        'num_entries',
        'top_seed_is_record_holder',
        'is_sprint', 'is_distance',
        'stroke_encoded',
        'season_peak',
        'num_sub_1pct_off_record',
        'top_3_spread'
    ]
    
    # Add positional features
    for pos in [1, 3, 5, 8]:
        feature_cols.extend([
            f'pos_{pos}_pct_off_world',
            f'pos_{pos}_pct_off_american',
            f'pos_{pos}_pct_off_us_open'
        ])
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[f'{target_record}_record_broken']
    
    return X, y

def train_evaluate_model(X, y, model_type='rf'):
    """Train and evaluate model with handling for extremely imbalanced data."""
    # Print initial class distribution
    print("\nInitial class distribution:")
    print(y.value_counts())
    
    # Check if we have enough samples of each class
    if len(y.unique()) < 2:
        print("\nWarning: Only one class present in the dataset.")
        return None, "Insufficient class diversity for modeling", {}
    
    if y.value_counts().min() < 2:
        print("\nWarning: Using simplified model due to extreme class imbalance")
        # Use a simpler pipeline without SMOTE
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=1000,  # Increased trees
                max_depth=3,        # Limit depth to prevent overfitting
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight={0: 1, 1: 100},  # Heavy weight on minority class
                random_state=42
            ))
        ])
    else:
        # Original pipeline with SMOTE for cases with enough samples
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, sampling_strategy=0.1)),
            ('classifier', RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ))
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    
    # Calculate additional metrics
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROC curve
    ax1.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    
    # Precision-Recall curve
    ax2.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"model_performance_curves.png")
    plt.close()
    
    return (
        pipeline,
        classification_report(y_test, y_pred, zero_division=0),
        {'roc_auc': roc_auc, 'avg_precision': avg_precision}
    )

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importances.sort_values('importance', ascending=False),
        x='importance',
        y='feature'
    )
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "feature_importance.png")

def main():
    # Load data
    print("Loading features...")
    df = load_features()
    
    # Train models for each record type
    for record_type in ['world', 'american', 'us_open']:
        print(f"\nTraining model for {record_type} records:")
        X, y = prepare_data(df, record_type)
        
        print(f"\nTotal samples: {len(y)}")
        print("Class distribution:")
        print(y.value_counts(normalize=True))
        
        pipeline, report, metrics = train_evaluate_model(X, y)
        
        print("\nClassification Report:")
        print(report)
        print("\nAdditional Metrics:")
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
        print(f"Average Precision: {metrics['avg_precision']:.3f}")
        
        plot_feature_importance(pipeline.named_steps['classifier'], X.columns)

if __name__ == "__main__":
    main()