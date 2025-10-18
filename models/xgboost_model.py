"""
XGBoost Model Wrapper for Trading Signals Prediction

Provides a unified interface for XGBoost classifier with:
- Automatic data type handling (numpy/torch)
- Model persistence (save/load)
- Feature importance analysis
- Early stopping support
"""

import numpy as np
import torch
import xgboost as xgb
import joblib
from pathlib import Path


class XGBoostModel:
    """
    XGBoost classifier wrapper for binary classification
    
    Features:
    - Handles both numpy arrays and torch tensors
    - Built-in early stopping
    - Feature importance tracking
    - Model serialization
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=None,
        random_state=42,
        early_stopping_rounds=20
    ):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds (trees)
            max_depth: Maximum tree depth (3-10 recommended)
            learning_rate: Step size shrinkage (0.01-0.3)
            min_child_weight: Minimum sum of instance weight in child (1-10)
            subsample: Row sampling ratio (0.5-1.0)
            colsample_bytree: Column sampling ratio (0.5-1.0)
            scale_pos_weight: Balancing weight for positive class
            random_state: Random seed for reproducibility
            early_stopping_rounds: Rounds without improvement before stop
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        
        # Model instance (initialized on fit)
        self.model = None
        self.best_iteration = None
        
    def _convert_to_numpy(self, X):
        """Convert input to numpy array"""
        if isinstance(X, torch.Tensor):
            return X.cpu().numpy()
        return np.asarray(X)
    
    def fit(self, X, y, eval_set=None, verbose=True):
        """
        Train XGBoost model
        
        Args:
            X: Features, shape (N, 31), numpy array or torch tensor
            y: Labels, shape (N,), binary {0, 1}
            eval_set: Tuple (X_val, y_val) for validation and early stopping
            verbose: Whether to print training progress
            
        Returns:
            self (for method chaining)
        """
        # Convert inputs
        X_train = self._convert_to_numpy(X)
        y_train = self._convert_to_numpy(y)
        
        # Build eval set
        eval_set_xgb = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = self._convert_to_numpy(X_val)
            y_val = self._convert_to_numpy(y_val)
            # FIXED: Proper eval_set format for XGBoost with both train and val
            eval_set_xgb = [(X_train, y_train), (X_val, y_val)]
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=self.early_stopping_rounds if eval_set_xgb else None
        )
        
        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set_xgb,
            verbose=verbose
        )
        
        # Store best iteration
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration = self.model.best_iteration
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features, shape (N, 31)
            
        Returns:
            Probabilities array, shape (N, 2)
            Column 0: P(class=0) [DOWN]
            Column 1: P(class=1) [UP]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_pred = self._convert_to_numpy(X)
        return self.model.predict_proba(X_pred)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels with custom threshold
        
        Args:
            X: Features, shape (N, 31)
            threshold: Decision threshold for class 1 (default: 0.5)
            
        Returns:
            Predicted classes, shape (N,), values {0, 1}
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def save(self, path):
        """
        Save model to disk
        
        Args:
            path: File path (e.g., 'models/xgboost_model.json')
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path))
        
        # Save metadata
        metadata = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': self.random_state,
            'early_stopping_rounds': self.early_stopping_rounds,
            'best_iteration': self.best_iteration
        }
        
        metadata_path = path.with_suffix('.metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"✅ Model saved to {path}")
        print(f"✅ Metadata saved to {metadata_path}")
    
    def load(self, path):
        """
        Load model from disk
        
        Args:
            path: File path to model file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load metadata
        metadata_path = path.with_suffix('.metadata.pkl')
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.__dict__.update(metadata)
        
        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        
        print(f"✅ Model loaded from {path}")
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance scores
        
        Args:
            importance_type: Type of importance
                - 'gain': Average gain across all splits
                - 'weight': Number of times feature appears in trees
                - 'cover': Average coverage of splits
                
        Returns:
            dict: {feature_index: importance_score}
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert feature names (f0, f1, ...) to indices
        importance_dict = {}
        for feat_name, score in importance.items():
            feat_idx = int(feat_name.replace('f', ''))
            importance_dict[feat_idx] = score
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def print_feature_importance(self, top_n=10):
        """
        Print top N most important features
        
        Args:
            top_n: Number of features to display
        """
        importance = self.get_feature_importance()
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} FEATURE IMPORTANCE (GAIN)")
        print(f"{'='*60}")
        
        for i, (feat_idx, score) in enumerate(list(importance.items())[:top_n], 1):
            print(f"{i:2d}. Feature {feat_idx:2d}: {score:8.2f}")
        
        print(f"{'='*60}\n")
    
    def __repr__(self):
        """String representation"""
        if self.model is None:
            status = "NOT TRAINED"
        else:
            status = f"TRAINED (best_iter={self.best_iteration})"
        
        return (
            f"XGBoostModel(\n"
            f"  status={status},\n"
            f"  n_estimators={self.n_estimators},\n"
            f"  max_depth={self.max_depth},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  scale_pos_weight={self.scale_pos_weight}\n"
            f")"
        )