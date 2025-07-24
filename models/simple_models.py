"""
Simplified Machine Learning Models for ProphitBet
Using only scikit-learn without TensorFlow to avoid disk quota issues
"""

import os
import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import joblib

logger = logging.getLogger(__name__)


class SimpleRandomForestModel:
    """Simplified Random Forest model for football prediction"""
    
    def __init__(self, model_name: str = "simple_rf"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
        # Default hyperparameters
        self.hyperparameters = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    
    def create_model(self, hyperparameters: Dict[str, Any] = None) -> RandomForestClassifier:
        """Create Random Forest model with given hyperparameters"""
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        
        # Create base Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.hyperparameters['n_estimators'],
            max_depth=self.hyperparameters['max_depth'],
            min_samples_split=self.hyperparameters['min_samples_split'],
            min_samples_leaf=self.hyperparameters['min_samples_leaf'],
            random_state=self.hyperparameters['random_state'],
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all processors
        )
        
        # Wrap with calibration for probability estimates
        self.model = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: list = None) -> Dict[str, Any]:
        """Train the Random Forest model"""
        try:
            if self.model is None:
                self.create_model()
            
            # Store feature names
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            logger.info(f"Training {self.model_name} with {X_train.shape[0]} samples...")
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Evaluate on validation set if provided
            results = {}
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                val_predictions = self.model.predict(X_val_scaled)
                val_probabilities = self.model.predict_proba(X_val_scaled)
                
                results.update({
                    'val_accuracy': accuracy_score(y_val, val_predictions),
                    'val_predictions': val_predictions,
                    'val_probabilities': val_probabilities,
                    'val_report': classification_report(y_val, val_predictions, output_dict=True)
                })
            
            # Training accuracy
            train_predictions = self.model.predict(X_train_scaled)
            results.update({
                'train_accuracy': accuracy_score(y_train, train_predictions),
                'feature_importance': self._get_feature_importance(),
                'model_name': self.model_name,
                'hyperparameters': self.hyperparameters.copy()
            })
            
            logger.info(f"Model training completed. Train accuracy: {results['train_accuracy']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with the trained model"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test set"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            predictions, probabilities = self.predict(X_test)
            
            return {
                'accuracy': accuracy_score(y_test, predictions),
                'classification_report': classification_report(y_test, predictions, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
                'predictions': predictions,
                'probabilities': probabilities
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save(self, filepath: str) -> bool:
        """Save the trained model"""
        try:
            if not self.is_trained:
                logger.warning("Attempting to save untrained model")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'hyperparameters': self.hyperparameters,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.hyperparameters = model_data['hyperparameters']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            self.model_name = model_data['model_name']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained model"""
        try:
            if not self.is_trained or self.model is None:
                return None
            
            # Get feature importance from the base Random Forest
            base_model = self.model.base_estimator
            if hasattr(base_model, 'feature_importances_'):
                importance_dict = {}
                for i, importance in enumerate(base_model.feature_importances_):
                    feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
                    importance_dict[feature_name] = float(importance)
                return importance_dict
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None


class SimpleMLPModel:
    """Simplified Multi-Layer Perceptron using scikit-learn"""
    
    def __init__(self, model_name: str = "simple_mlp"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
        # Default hyperparameters
        self.hyperparameters = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'random_state': 42
        }
    
    def create_model(self, hyperparameters: Dict[str, Any] = None):
        """Create MLP model with given hyperparameters"""
        from sklearn.neural_network import MLPClassifier
        
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hyperparameters['hidden_layer_sizes'],
            activation=self.hyperparameters['activation'],
            solver=self.hyperparameters['solver'],
            alpha=self.hyperparameters['alpha'],
            batch_size=self.hyperparameters['batch_size'],
            learning_rate=self.hyperparameters['learning_rate'],
            learning_rate_init=self.hyperparameters['learning_rate_init'],
            max_iter=self.hyperparameters['max_iter'],
            random_state=self.hyperparameters['random_state']
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: list = None) -> Dict[str, Any]:
        """Train the MLP model"""
        try:
            if self.model is None:
                self.create_model()
            
            # Store feature names
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            logger.info(f"Training {self.model_name} with {X_train.shape[0]} samples...")
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Evaluate on validation set if provided
            results = {}
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                val_predictions = self.model.predict(X_val_scaled)
                val_probabilities = self.model.predict_proba(X_val_scaled)
                
                results.update({
                    'val_accuracy': accuracy_score(y_val, val_predictions),
                    'val_predictions': val_predictions,
                    'val_probabilities': val_probabilities,
                    'val_report': classification_report(y_val, val_predictions, output_dict=True)
                })
            
            # Training accuracy
            train_predictions = self.model.predict(X_train_scaled)
            results.update({
                'train_accuracy': accuracy_score(y_train, train_predictions),
                'model_name': self.model_name,
                'hyperparameters': self.hyperparameters.copy()
            })
            
            logger.info(f"Model training completed. Train accuracy: {results['train_accuracy']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with the trained model"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save(self, filepath: str) -> bool:
        """Save the trained model"""
        try:
            if not self.is_trained:
                logger.warning("Attempting to save untrained model")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'hyperparameters': self.hyperparameters,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.hyperparameters = model_data['hyperparameters']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            self.model_name = model_data['model_name']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False