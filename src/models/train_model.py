import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
    def compare_models(self, X, y, cv=5):
        """Compare baseline models with cross-validation."""
        print(" Comparing models...")
        
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
            self.results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
            
            if scores.mean() > self.best_score:
                self.best_score = scores.mean()
                self.best_model_name = name
                self.best_model = model
                
        print(f"\n Best baseline: {self.best_model_name} ({self.best_score:.4f})")
        return self.results
    
    def optimize_hyperparams(self, X, y):
        """Grid search for best hyperparameters."""
        print(f"\n Optimizing {self.best_model_name}...")
        
        if self.best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            model = RandomForestClassifier(random_state=42)
            
        elif self.best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=5, scoring='f1', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_model = grid_search.best_estimator_
        self.best_score = grid_search.best_score_
        
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV F1: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train_final(self, X, y):
        """Train final model on full dataset."""
        self.best_model.fit(X, y)
        return self.best_model
    
    def save_model(self, filepath='models/best_model.joblib'):
        """Save model and metadata."""
        joblib.dump(self.best_model, filepath)
        
        metadata = {
            'model_type': type(self.best_model).__name__,
            'best_cv_score': float(self.best_score),
            'timestamp': datetime.now().isoformat(),
            'parameters': self.best_model.get_params()
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Model saved to {filepath}")

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from src.data.load_data import load_raw_data
    from src.features.build_features import FeatureEngineer
    
    # Load & process
    train, _ = load_raw_data()
    fe = FeatureEngineer()
    X, y, _ = fe.fit_transform(train)
    
    # # Train
    trainer = ModelTrainer()
    trainer.compare_models(X, y)
    trainer.optimize_hyperparams(X, y)
    trainer.train_final(X, y)
    # trainer.save_model()