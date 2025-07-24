"""
Fully Connected Neural Network (FCNet) model for football prediction
"""

import numpy as np
import logging
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import optuna

logger = logging.getLogger(__name__)

class FCNet:
    """Fully Connected Neural Network for football match prediction"""
    
    def __init__(self, input_dim: int, num_classes: int = 3):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Optional[tf.keras.Model]:
        """Train the FCNet model with hyperparameter optimization"""
        try:
            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            # Convert labels to categorical
            y_train_cat = to_categorical(y_train_balanced, num_classes=self.num_classes)
            y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
            
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
                n_layers = trial.suggest_int('n_layers', 2, 5)
                units_base = trial.suggest_int('units_base', 64, 512)
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                
                # Build model
                model = self._build_model(
                    n_layers=n_layers,
                    units_base=units_base,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    l2_reg=l2_reg
                )
                
                # Train model
                history = model.fit(
                    X_train_balanced, y_train_cat,
                    batch_size=batch_size,
                    epochs=50,
                    validation_data=(X_val, y_val_cat),
                    callbacks=[
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(patience=5, factor=0.5)
                    ],
                    class_weight=class_weight_dict,
                    verbose=0
                )
                
                # Return validation accuracy
                return max(history.history['val_accuracy'])
            
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20, timeout=1800)  # 30 minutes max
            
            # Train final model with best parameters
            best_params = study.best_params
            logger.info(f"Best FCNet parameters: {best_params}")
            
            self.model = self._build_model(
                n_layers=best_params['n_layers'],
                units_base=best_params['units_base'],
                dropout_rate=best_params['dropout_rate'],
                learning_rate=best_params['learning_rate'],
                l2_reg=best_params['l2_reg']
            )
            
            # Train final model
            history = self.model.fit(
                X_train_balanced, y_train_cat,
                batch_size=best_params['batch_size'],
                epochs=100,
                validation_data=(X_val, y_val_cat),
                callbacks=[
                    EarlyStopping(patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(patience=7, factor=0.5)
                ],
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Evaluate final model
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val_cat, verbose=0)
            logger.info(f"FCNet final validation accuracy: {val_accuracy:.4f}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training FCNet model: {str(e)}")
            return None
    
    def _build_model(self, n_layers: int, units_base: int, dropout_rate: float,
                     learning_rate: float, l2_reg: float) -> tf.keras.Model:
        """Build the neural network architecture"""
        model = Sequential()
        
        # Input layer with noise for regularization
        model.add(GaussianNoise(0.1, input_shape=(self.input_dim,)))
        
        # Hidden layers
        for i in range(n_layers):
            units = max(units_base // (2 ** i), 32)  # Decrease units in each layer
            
            model.add(Dense(
                units,
                activation='relu',
                kernel_regularizer=l2(l2_reg),
                bias_regularizer=l2(l2_reg)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=l2_reg)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
