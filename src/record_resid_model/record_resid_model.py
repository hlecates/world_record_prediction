import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from record_resid_features import (
    load_and_split_event_files,
    merge_event_and_entries,
    engineer_model_features
)


def train_residual_model():
    """Train a model to predict gap to record time."""
    # Load and process data
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "processed" / "parsed_meet_results"
    
    # Load data for World records
    df_events, df_entries = load_and_split_event_files(data_dir, record_type="World")
    
    # Merge event and entry data
    df_merged = merge_event_and_entries(df_events, df_entries)
    
    # Engineer features
    X, y, df = engineer_model_features(df_merged)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"RMSE: {rmse:.2f} seconds")
    print(f"R2 Score: {r2:.3f}")
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance['importance'])
    plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=45, ha='right')
    plt.title('Feature Importance in Predicting Gap to Record')
    plt.tight_layout()
    
    # Save plot
    output_dir = base_dir / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()
    
    return model, feature_importance

if __name__ == "__main__":
    model, importance = train_residual_model()