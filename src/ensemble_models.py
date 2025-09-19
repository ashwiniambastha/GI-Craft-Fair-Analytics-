"""
Advanced Ensemble Machine Learning Models
========================================
7-model ensemble with meta-learning for vendor performance prediction
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleModel:
    """Advanced ensemble system with multiple base models and meta-learning"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_weights = {}
        
    def initialize_base_models(self):
        """Initialize all base models with optimized parameters"""
        
        # XGBoost
        self.base_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # LightGBM
        self.base_models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        # CatBoost
        self.base_models['catboost'] = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
        
        # Random Forest
        self.base_models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Extra Trees
        self.base_models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Neural Network
        self.base_models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=300,
            random_state=42
        )
        
        # Logistic Regression
        self.base_models['logistic'] = LogisticRegression(
            C=1.0,
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            max_iter=1000,
            random_state=42
        )
        
        print(f"Initialized {len(self.base_models)} base models")
    
    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """Train all base models and collect validation predictions"""
        
        print("Training base models...")
        base_predictions = {}
        model_scores = {}
        
        # Scale features for neural network and logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            
            try:
                # Use scaled data for models that need it
                if name in ['neural_network', 'logistic']:
                    model.fit(X_train_scaled, y_train)
                    val_pred = model.predict(X_val_scaled)
                    val_pred_proba = model.predict_proba(X_val_scaled)
                else:
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val)
                    val_pred_proba = model.predict_proba(X_val)
                
                # Store predictions and scores
                base_predictions[name] = val_pred_proba
                model_scores[name] = {
                    'accuracy': accuracy_score(y_val, val_pred),
                    'roc_auc': roc_auc_score(y_val, val_pred_proba, multi_class='ovr', average='weighted')
                }
                
                # Store feature importance (where available)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    # For logistic regression, use absolute coefficients
                    if len(model.coef_.shape) > 1:
                        coef = np.mean(np.abs(model.coef_), axis=0)
                    else:
                        coef = np.abs(model.coef_[0])
                    self.feature_importance[name] = dict(zip(X_train.columns, coef))
                
                print(f"{name} - Accuracy: {model_scores[name]['accuracy']:.4f}, ROC-AUC: {model_scores[name]['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        return base_predictions, model_scores
    
    def train_meta_model(self, base_predictions: dict, y_val: pd.Series):
        """Train meta-model on base model predictions"""
        print("Training meta-model...")
        
        # Create meta-features from base model predictions
        meta_features = []
        for name, preds in base_predictions.items():
            # Use class probabilities as meta-features
            for class_idx in range(preds.shape[1]):
                meta_features.append(preds[:, class_idx])
        
        meta_X = np.column_stack(meta_features)
        
        # Train meta-model (Ridge regression for stability)
        self.meta_model = Ridge(alpha=1.0, random_state=42)
        self.meta_model.fit(meta_X, y_val)
        
        # Calculate model weights based on meta-model coefficients
        n_models = len(base_predictions)
        n_classes = list(base_predictions.values())[0].shape[1]
        
        model_weights = {}
        coef_idx = 0
        for name in base_predictions.keys():
            # Average absolute coefficients for this model across all classes
            model_coef = []
            for class_idx in range(n_classes):
                model_coef.append(np.abs(self.meta_model.coef_[coef_idx]))
                coef_idx += 1
            model_weights[name] = np.mean(model_coef)
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        self.model_weights = {name: weight/total_weight for name, weight in model_weights.items()}
        
        print("Meta-model weights:")
        for name, weight in self.model_weights.items():
            print(f"  {name}: {weight:.4f}")
    
    def predict_ensemble(self, X_test: pd.DataFrame) -> tuple:
        """Make ensemble predictions using trained models"""
        
        # Get predictions from all base models
        base_predictions = {}
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.base_models.items():
            if name in ['neural_network', 'logistic']:
                pred_proba = model.predict_proba(X_test_scaled)
            else:
                pred_proba = model.predict_proba(X_test)
            base_predictions[name] = pred_proba
        
        # Create meta-features
        meta_features = []
        for name, preds in base_predictions.items():
            for class_idx in range(preds.shape[1]):
                meta_features.append(preds[:, class_idx])
        
        meta_X = np.column_stack(meta_features)
        
        # Meta-model prediction
        meta_pred = self.meta_model.predict(meta_X)
        final_pred = np.round(meta_pred).astype(int)
        
        # Also compute weighted average as alternative
        weighted_pred_proba = np.zeros_like(list(base_predictions.values())[0])
        for name, pred_proba in base_predictions.items():
            weighted_pred_proba += self.model_weights[name] * pred_proba
        
        weighted_pred = np.argmax(weighted_pred_proba, axis=1)
        
        return final_pred, weighted_pred, weighted_pred_proba, base_predictions
    
    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray, model_name: str = "Ensemble"):
        """Comprehensive model evaluation"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_name.lower()}_confusion_matrix.png', dpi=300)
        plt.show()
        
        return {'accuracy': accuracy, 'roc_auc': roc_auc, 'confusion_matrix': cm}
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot aggregated feature importance across models"""
        
        if not self.feature_importance:
            print("No feature importance data available")
            return
        
        # Aggregate feature importance across models
        all_features = set()
        for model_features in self.feature_importance.values():
            all_features.update(model_features.keys())
        
        aggregated_importance = {}
        for feature in all_features:
            importance_values = []
            for model_name, model_features in self.feature_importance.items():
                if feature in model_features:
                    # Weight by model performance weight
                    weighted_importance = model_features[feature] * self.model_weights.get(model_name, 1.0)
                    importance_values.append(weighted_importance)
            
            if importance_values:
                aggregated_importance[feature] = np.mean(importance_values)
        
        # Sort and plot top features
        sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importance_values = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importance_values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Aggregated Feature Importance')
        plt.title(f'Top {top_n} Most Important Features (Ensemble)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('results/plots/ensemble_feature_importance.png', dpi=300)
        plt.show()
        
        return dict(sorted_features)
    
    def cross_validate_ensemble(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
        """Cross-validate the ensemble model"""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = {name: [] for name in self.base_models.keys()}
        cv_scores['ensemble'] = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{cv_folds}")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Initialize fresh models for this fold
            self.initialize_base_models()
            
            # Train base models
            base_preds, model_scores = self.train_base_models(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # Store individual model scores
            for name, scores in model_scores.items():
                cv_scores[name].append(scores['accuracy'])
            
            # Train meta-model
            self.train_meta_model(base_preds, y_val_fold)
            
            # Ensemble prediction
            ensemble_pred, _, _, _ = self.predict_ensemble(X_val_fold)
            ensemble_accuracy = accuracy_score(y_val_fold, ensemble_pred)
            cv_scores['ensemble'].append(ensemble_accuracy)
        
        # Calculate mean and std for each model
        cv_results = {}
        for model_name, scores in cv_scores.items():
            cv_results[model_name] = {
                'mean_accuracy': np.mean(scores),
                'std_accuracy': np.std(scores),
                'scores': scores
            }
        
        print("\nCross-Validation Results:")
        for model_name, results in cv_results.items():
            print(f"{model_name}: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        
        return cv_results
    
    def save_models(self, model_name: str = "ensemble_model"):
        """Save all trained models"""
        
        # Save base models
        for name, model in self.base_models.items():
            joblib.dump(model, f'models/{model_name}_{name}.pkl')
        
        # Save meta-model and scaler
        joblib.dump(self.meta_model, f'models/{model_name}_meta.pkl')
        joblib.dump(self.scaler, f'models/{model_name}_scaler.pkl')
        
        # Save model weights and feature importance
        import json
        with open(f'models/{model_name}_weights.json', 'w') as f:
            json.dump(self.model_weights, f, indent=2)
        
        with open(f'models/{model_name}_feature_importance.json', 'w') as f:
            json.dump(self.feature_importance, f, indent=2, default=str)
        
        print(f"All models saved with prefix: {model_name}")
    
    def load_models(self, model_name: str = "ensemble_model"):
        """Load pre-trained models"""
        
        # Load base models
        model_files = [
            'xgboost', 'lightgbm', 'catboost', 'random_forest', 
            'extra_trees', 'neural_network', 'logistic'
        ]
        
        for name in model_files:
            try:
                self.base_models[name] = joblib.load(f'models/{model_name}_{name}.pkl')
            except FileNotFoundError:
                print(f"Model file not found: {model_name}_{name}.pkl")
        
        # Load meta-model and scaler
        self.meta_model = joblib.load(f'models/{model_name}_meta.pkl')
        self.scaler = joblib.load(f'models/{model_name}_scaler.pkl')
        
        # Load weights and feature importance
        import json
        with open(f'models/{model_name}_weights.json', 'r') as f:
            self.model_weights = json.load(f)
        
        with open(f'models/{model_name}_feature_importance.json', 'r') as f:
            self.feature_importance = json.load(f)
        
        print(f"Models loaded successfully")

def main():
    """Run complete ensemble modeling pipeline"""
    
    # Load engineered features data
    try:
        df = pd.read_csv("data/processed/engineered_features.csv")
    except FileNotFoundError:
        print("Engineered features not found. Using processed data...")
        df = pd.read_csv("data/processed/processed_data.csv")
    
    # Prepare features and target
    target_col = 'Performance_Category'
    
    # Select features (exclude ID columns and target)
    exclude_cols = ['Stall_ID', 'Performance_Category', 'Churned', 'Still_Active']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle categorical columns by selecting only numeric features
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[numeric_cols].fillna(0)  # Handle any remaining NaNs
    y = df[target_col]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize ensemble model
    ensemble = AdvancedEnsembleModel()
    ensemble.initialize_base_models()
    
    # Train base models
    base_predictions, model_scores = ensemble.train_base_models(X_train, y_train, X_val, y_val)
    
    # Train meta-model
    ensemble.train_meta_model(base_predictions, y_val)
    
    # Make predictions on test set
    final_pred, weighted_pred, weighted_pred_proba, _ = ensemble.predict_ensemble(X_test)
    
    # Evaluate ensemble model
    ensemble_results = ensemble.evaluate_model(y_test, final_pred, weighted_pred_proba, "Ensemble")
    
    # Also evaluate weighted average approach
    weighted_results = ensemble.evaluate_model(y_test, weighted_pred, weighted_pred_proba, "Weighted_Average")
    
    # Plot feature importance
    top_features = ensemble.plot_feature_importance(top_n=20)
    
    # Cross-validation
    cv_results = ensemble.cross_validate_ensemble(X, y, cv_folds=5)
    
    # Save models
    ensemble.save_models("gi_craft_ensemble")
    
    # Save results
    results_summary = {
        'test_accuracy': ensemble_results['accuracy'],
        'test_roc_auc': ensemble_results['roc_auc'],
        'cv_results': cv_results,
        'model_weights': ensemble.model_weights,
        'top_features': top_features
    }
    
    import json
    with open('results/ensemble_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Ensemble_Prediction': final_pred,
        'Weighted_Prediction': weighted_pred
    })
    
    # Add probability scores
    for i, class_name in enumerate(['Low_Performer', 'Medium_Performer', 'High_Performer']):
        predictions_df[f'Prob_{class_name}'] = weighted_pred_proba[:, i]
    
    predictions_df.to_csv('results/ensemble_predictions.csv', index=False)
    
    print(f"\nFinal Results:")
    print(f"Ensemble Test Accuracy: {ensemble_results['accuracy']:.4f}")
    print(f"Ensemble Test ROC-AUC: {ensemble_results['roc_auc']:.4f}")
    print(f"Cross-Validation Accuracy: {cv_results['ensemble']['mean_accuracy']:.4f} ± {cv_results['ensemble']['std_accuracy']:.4f}")
    
    return ensemble, results_summary, predictions_df

if __name__ == "__main__":
    ensemble_model, results, predictions = main()
    print("Ensemble modeling completed successfully!")
