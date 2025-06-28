import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_auc_score, roc_curve)


class Preprocessor:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}

    
    def load_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        return df
    

    def prepare_features_targets(self, df: pd.DataFrame) -> Dict[str, Tuple]:
        targets = {
            'world_record_residual': 'winner_vs_world_record',
            'american_record_residual': 'winner_vs_american_record', 
            'us_open_record_residual': 'winner_vs_us_open_record',
            'top_seed_win': 'top_seed_won'
        }

        # Define features to exclude, being the targets and identifying info
        exclude_features = [
            'meet', 'event_type', 'winner_vs_world_record', 'winner_vs_american_record',
            'winner_vs_us_open_record', 'top_seed_won', 'stroke'
        ]

        # Get feature columns
        feature_columns = [col for col in df.columns if col not in exclude_features]
        
        prepared_data = {}

        for task_name, target_col in targets.items():
            task_df = df.dropna(subset=[target_col]).copy()

            X = task_df[feature_columns].copy()
            y = task_df[target_col].copy()

            X = self._encode_categorical_features(X, task_name)

            X = self._handle_missing_values(X)

            X_selected = self._select_features(X, y, task_name, is_classification=(task_name == 'top_seed'))

            X_scaled = self._scale_features(X_selected, task_name)

            prepared_data[task_name] = (X_scaled, y, X_selected.columns.tolist())

        return prepared_data
    

    def _encode_categorical_features(self, X: pd.DataFrame, task_name: str) -> pd.DataFrame:
        non_numeric = X.select_dtypes(exclude=[np.number, bool]).columns.tolist()
        print(f"[{task_name}] Non-numeric columns: {non_numeric}")


        categorical_cols = ['distance_category', 'stroke_category', 'gender', ]

        for col in categorical_cols:
            if col in X.columns:
                if task_name not in self.label_encoders:
                    self.label_encoders[task_name] = {}

                if col not in self.label_encoders[task_name]:
                    self.label_encoders[task_name][col] = LabelEncoder()
                    X[col] = self.label_encoders[task_name][col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[task_name][col].transform(X[col].astype(str))

        return X
    

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        # Fill numeric missing values with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        
        # Fill boolean missing values with False
        bool_cols = X.select_dtypes(include=[bool]).columns
        for col in bool_cols:
            X[col] = X[col].fillna(False)
        
        return X
    

    def _select_features(self, X: pd.DataFrame, y: pd.Series, task_name: str, is_classification: bool=False) -> pd.DataFrame:
        score_func = f_classif if is_classification else f_regression

        # Select top features based on univariate statistical tests
        n_features = min(50, X.shape[1])
        
        selector = SelectKBest(score_func=score_func, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_selectors[task_name] = {
            'selector': selector,
            'features': selected_features
        }
        
        print(f"Selected {len(selected_features)} features for {task_name}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    

    def _scale_features(self, X: pd.DataFrame, task_name: str) -> np.ndarray:
        if task_name not in self.scalers:
            self.scalers[task_name] = StandardScaler()
            X_scaled = self.scalers[task_name].fit_transform(X)
        else:
            X_scaled = self.scalers[task_name].transform(X)
        
        return X_scaled
    

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}


    def define_model_configurations(self) -> Dict[str, Dict]:
        regression_models = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
            },
            'lasso': {
                'model': Lasso(),
                'params': {'alpha': [0.01, 0.1, 1.0, 10.0]}
            },
            'elastic_net': {
                'model': ElasticNet(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
        
        classification_models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {'C': [0.1, 1.0, 10.0, 100.0]}
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
        
        return {
            'regression': regression_models,
            'classification': classification_models
        }
    

    def train_models(self, prepared_data: Dict[str, Tuple]) -> Dict[str, Dict]:
        model_configs = self.define_model_configurations()
        
        for task_name, (X, y, feature_names) in prepared_data.items():
            print(f"Training models for {task_name}")
            
            # Determine if this is classification or regression
            is_classification = task_name == 'top_seed_win'
            model_type = 'classification' if is_classification else 'regression'
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if is_classification else None
            )
            
            self.models[task_name] = {}
            self.results[task_name] = {}
            
            # Train each model type
            for model_name, config in model_configs[model_type].items():
                print(f"\nTraining {model_name} for {task_name}...")
                
                # Grid search for best parameters
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if is_classification else KFold(n_splits=5, shuffle=True, random_state=42)
                
                if config['params']:
                    grid_search = GridSearchCV(
                        config['model'], 
                        config['params'],
                        cv=cv,
                        scoring='roc_auc' if is_classification else 'neg_mean_squared_error',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    print(f"Best parameters: {grid_search.best_params_}")
                else:
                    best_model = config['model']
                    best_model.fit(X_train, y_train)
                
                # Store model
                self.models[task_name][model_name] = {
                    'model': best_model,
                    'feature_names': feature_names
                }
                
                # Evaluate model
                self.results[task_name][model_name] = self._evaluate_model(
                    best_model, X_train, X_test, y_train, y_test, is_classification
                )
        
        return self.results
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test, is_classification: bool) -> Dict:
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        if is_classification:
            # Classification metrics
            y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
            y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            results = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'train_f1': f1_score(y_train, y_train_pred),
                'test_f1': f1_score(y_test, y_test_pred),
                'train_precision': precision_score(y_train, y_train_pred),
                'test_precision': precision_score(y_test, y_test_pred),
                'train_recall': recall_score(y_train, y_train_pred),
                'test_recall': recall_score(y_test, y_test_pred),
            }
            
            if y_test_proba is not None:
                results['train_auc'] = roc_auc_score(y_train, y_train_proba)
                results['test_auc'] = roc_auc_score(y_test, y_test_proba)
            
        else:
            # Regression metrics
            results = {
                'train_mse': mean_squared_error(y_train, y_train_pred),
                'test_mse': mean_squared_error(y_test, y_test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred)
            }
        
        return results
    

class ModelEvaluation:
    def __init__(self):
        pass

    
    def print_results_summary(self, results: Dict[str, Dict]):
        for task_name, task_results in results.items():
            print(f"\nRESULTS SUMMARY: {task_name.upper()}")
            
            # Determine if classification or regression
            is_classification = task_name == 'top_seed_win'
            
            if is_classification:
                # Classification results table
                print(f"{'Model':<20} {'Test Acc':<10} {'Test F1':<10} {'Test AUC':<10} {'Test Prec':<10} {'Test Rec':<10}")
                print("-" * 70)
                
                for model_name, metrics in task_results.items():
                    auc_str = f"{metrics.get('test_auc', 0):.3f}" if 'test_auc' in metrics else "N/A"
                    print(f"{model_name:<20} {metrics['test_accuracy']:<10.3f} {metrics['test_f1']:<10.3f} "
                          f"{auc_str:<10} {metrics['test_precision']:<10.3f} {metrics['test_recall']:<10.3f}")
            
            else:
                # Regression results table
                print(f"{'Model':<20} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<12} {'Train R²':<12}")
                print("-" * 68)
                
                for model_name, metrics in task_results.items():
                    print(f"{model_name:<20} {metrics['test_rmse']:<12.3f} {metrics['test_mae']:<12.3f} "
                          f"{metrics['test_r2']:<12.3f} {metrics['train_r2']:<12.3f}")
    
    def identify_best_models(self, results: Dict[str, Dict]) -> Dict[str, str]:
        best_models = {}
        
        for task_name, task_results in results.items():
            if task_name == 'top_seed_win':
                # For classification, use test AUC if available, otherwise test F1
                metric_key = 'test_auc' if any('test_auc' in metrics for metrics in task_results.values()) else 'test_f1'
                best_model = max(task_results.items(), key=lambda x: x[1].get(metric_key, 0))
            else:
                # For regression, use test R²
                best_model = max(task_results.items(), key=lambda x: x[1].get('test_r2', -np.inf))
            
            best_models[task_name] = best_model[0]
            print(f"\nBest model for {task_name}: {best_model[0]}")
        
        return best_models

class ModelSaver:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_models_and_preprocessing(self, models: Dict, preprocessor: Preprocessor, 
                                    best_models: Dict[str, str]):
        
        print(f"\nSaving models to {self.output_dir}")
        
        # Save preprocessing objects
        preprocessing_objects = {
            'scalers': preprocessor.scalers,
            'label_encoders': preprocessor.label_encoders,
            'feature_selectors': preprocessor.feature_selectors
        }
        
        with open(self.output_dir / 'preprocessing.pkl', 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        
        # Save best models
        saved_models = {}
        for task_name, best_model_name in best_models.items():
            if task_name in models and best_model_name in models[task_name]:
                model_info = models[task_name][best_model_name]
                saved_models[task_name] = {
                    'model': model_info['model'],
                    'feature_names': model_info['feature_names'],
                    'model_type': best_model_name
                }
                print(f"Prepared {task_name}: {best_model_name}")
        
        with open(self.output_dir / 'best_models.pkl', 'wb') as f:
            pickle.dump(saved_models, f)
        
        # Save model metadata
        metadata = {
            'tasks': list(best_models.keys()),
            'best_models': best_models,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(self.output_dir / 'model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)


def main():
    # Configuration
    features_file = Path(__file__).parent.parent / "data" / "processed" / "features" / "features.csv"
    models_output_dir = Path(__file__).parent.parent / "output"/ "models"
    
    # Initialize components
    preprocessor = Preprocessor()
    trainer = ModelTrainer()
    evaluator = ModelEvaluation()
    saver = ModelSaver(models_output_dir)
    
    # Load and prepare data
    df = preprocessor.load_data(features_file)
    prepared_data = preprocessor.prepare_features_targets(df)
    
    if not prepared_data:
        print("No valid data prepared")
        return
    
    # Train models
    results = trainer.train_models(prepared_data)
    
    # Evaluate results
    evaluator.print_results_summary(results)
    best_models = evaluator.identify_best_models(results)
    
    # Save models
    saver.save_models_and_preprocessing(trainer.models, preprocessor, best_models)
    
    print(f"\nModels saved to: {models_output_dir}")

if __name__ == "__main__":
    main()




                

