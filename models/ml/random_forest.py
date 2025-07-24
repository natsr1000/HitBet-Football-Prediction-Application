"""
Random Forest model for football prediction
"""

import numpy as np
import logging
from typing import Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import optuna

logger = logging.getLogger(__name__)

class RandomForestModel:
    """Random Forest model for football match prediction"""
    
    def __init__(self):
        self.model = None
        self.is_calibrated = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Optional[RandomForestClassifier]:
        """Train the Random Forest model with hyperparameter optimization"""
        try:
            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_balanced),
                y=y_train_balanced
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            # Hyperparameter optimization with Optuna
            def objective(trial):
                # Suggest hyperparameters
                n_estimators = trial.suggest_int('n_estimators', 100, 1000)
                max_depth = trial.suggest_int('max_depth', 5, 30)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                
                # Build model
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    class_weight=class_weight_dict,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Train and evaluate
                model.fit(X_train_balanced, y_train_balanced)
                score = model.score(X_val, y_val)
                
                return score
            
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50, timeout=1800)  # 30 minutes max
            
            # Train final model with best parameters
            best_params = study.best_params
            logger.info(f"Best Random Forest parameters: {best_params}")
            
            self.model = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                max_features=best_params['max_features'],
                bootstrap=best_params['bootstrap'],
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            )
            
            # Train final model
            self.model.fit(X_train_balanced, y_train_balanced)
            
            # Calibrate the model for better probability estimates
            self.model = CalibratedClassifierCV(self.model, method='isotonic', cv=3)
            self.model.fit(X_train_balanced, y_train_balanced)
            self.is_calibrated = True
            
            # Evaluate final model
            val_accuracy = self.model.score(X_val, y_val)
            logger.info(f"Random Forest final validation accuracy: {val_accuracy:.4f}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.is_calibrated:
            # For calibrated classifier, get from base estimator
            return self.model.base_estimator.feature_importances_
        else:
            return self.model.feature_importances_
