"""
Service for handling machine learning model training and predictions
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
# import seaborn as sns  # Commented out to avoid dependency issues
from io import BytesIO
import base64

from repositories.model_repository import ModelRepository
from services.data_service import DataService
# from models.fcnet import FCNet  # Temporarily disabled - TensorFlow not available
# from models.random_forest import RandomForestModel  # Temporarily disabled
from models.simple_models import SimpleRandomForestModel, SimpleMLPModel
from utils.feature_engineering import compute_team_statistics
from utils.team_mapping import map_team_names

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for model training and predictions"""
    
    def __init__(self, model_repo: ModelRepository, data_service: DataService):
        self.model_repo = model_repo
        self.data_service = data_service
        self.scalers = {}  # Store scalers for each model
    
    def train_model(self, league_name: str, model_type: str, model_name: str) -> Tuple[bool, str]:
        """Train a new model"""
        try:
            # Prepare training data
            training_data = self.data_service.prepare_training_data(league_name)
            if training_data is None:
                return False, f"Failed to prepare training data for {league_name}"
            
            X, y = training_data
            
            if len(X) < 100:
                return False, f"Insufficient data for training: {len(X)} samples (minimum 100 required)"
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model
            if model_type == 'fcnet':
                # model = FCNet(input_dim=X_train.shape[1], num_classes=3)  # Disabled
                return False, "FCNet model temporarily disabled due to TensorFlow dependencies"
            elif model_type == 'random_forest':
                # model = RandomForestModel()  # Disabled
                model = SimpleRandomForestModel(model_name)
            elif model_type == 'simple_random_forest':
                model = SimpleRandomForestModel(model_name)
            elif model_type == 'simple_mlp':
                model = SimpleMLPModel(model_name)
            else:
                return False, f"Invalid model type: {model_type}. Available: random_forest, simple_random_forest, simple_mlp"
            
            # Train model
            logger.info(f"Training {model_type} model for {league_name}")
            
            # For simplified models, the training method returns results dict, not the model itself
            if model_type in ['simple_random_forest', 'simple_mlp']:
                training_results = model.train(X_train, y_train, X_test, y_test)
                if training_results is None:
                    return False, "Model training failed"
                
                # Get predictions for evaluation
                y_pred_classes, _ = model.predict(X_test)
                trained_model = model  # The model object itself
                
            else:
                # Legacy model training (for backward compatibility)
                trained_model = model.train(X_train_scaled, y_train, X_test_scaled, y_test)
                if trained_model is None:
                    return False, "Model training failed"
                
                # Evaluate model
                if model_type == 'fcnet':
                    y_pred = trained_model.predict(X_test_scaled)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                else:
                    y_pred_classes = trained_model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred_classes)
            
            # Calculate class distribution
            unique, counts = np.unique(y, return_counts=True)
            class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
            
            # Prepare metadata
            metadata = {
                'accuracy': float(accuracy),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': int(X_train.shape[1]),
                'class_distribution': class_distribution,
                'test_report': classification_report(y_test, y_pred_classes, output_dict=True)
            }
            
            # Save model and scaler
            success = self.model_repo.save_model(model_name, trained_model, model_type, league_name, metadata)
            
            if success:
                # Save scaler separately
                self.model_repo.save_scaler(model_name, scaler)
                message = f"Successfully trained {model_type} model '{model_name}' with accuracy: {accuracy:.3f}"
                return True, message
            else:
                return False, "Failed to save trained model"
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False, f"Training failed: {str(e)}"
    
    def predict_single_match(self, model_name: str, home_team: str, away_team: str) -> Optional[Dict]:
        """Make prediction for a single match"""
        try:
            # Load model and metadata
            model = self.model_repo.load_model(model_name)
            metadata = self.model_repo.get_model_metadata(model_name)
            
            if model is None or metadata is None:
                logger.error(f"Failed to load model {model_name} or its metadata")
                return None
            
            # For simplified models, scaler might not be needed
            scaler = self.model_repo.load_scaler(model_name)
            model_type = metadata['model_type']
            
            league_name = metadata['league_name']
            
            # Load league data for team statistics
            league_data = self.data_service.league_repo.load_league_data(league_name)
            if league_data is None:
                logger.error(f"League data not found for {league_name}")
                return None
            
            # Map team names
            mapped_home = map_team_names(home_team, league_data)
            mapped_away = map_team_names(away_team, league_data)
            
            if mapped_home is None or mapped_away is None:
                return {
                    'error': f"Teams not found in {league_name} data",
                    'suggested_teams': sorted(set(league_data['HomeTeam'].unique()) | set(league_data['AwayTeam'].unique()))
                }
            
            # Compute team statistics
            home_stats = compute_team_statistics(league_data, mapped_home)
            away_stats = compute_team_statistics(league_data, mapped_away)
            
            if home_stats is None or away_stats is None:
                logger.error("Failed to compute team statistics")
                return None
            
            # Prepare features
            features = np.concatenate([home_stats, away_stats]).reshape(1, -1)
            
            # Make prediction based on model type
            if model_type in ['simple_random_forest', 'simple_mlp']:
                # For simplified models, use the model's own predict method
                prediction, probabilities = model.predict(features)
                prediction = prediction[0] if isinstance(prediction, np.ndarray) else prediction
                prediction_probs = probabilities[0] if isinstance(probabilities, np.ndarray) else probabilities
            elif model_type == 'fcnet':
                # Scale features for FCNet
                if scaler is not None:
                    features_scaled = scaler.transform(features)
                else:
                    features_scaled = features
                prediction_probs = model.predict(features_scaled)[0]
                prediction = np.argmax(prediction_probs)
            else:
                # For other models
                if scaler is not None:
                    features_scaled = scaler.transform(features)
                else:
                    features_scaled = features
                prediction = model.predict(features_scaled)[0]
                prediction_probs = model.predict_proba(features_scaled)[0]
            
            # Format results
            outcomes = ['Home Win', 'Draw', 'Away Win']
            
            result = {
                'home_team': mapped_home,
                'away_team': mapped_away,
                'prediction': outcomes[prediction],
                'confidence': float(prediction_probs[prediction]),
                'probabilities': {
                    'home_win': float(prediction_probs[0]),
                    'draw': float(prediction_probs[1]),
                    'away_win': float(prediction_probs[2])
                },
                'model_name': model_name,
                'model_accuracy': metadata.get('accuracy', 0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making single prediction: {str(e)}")
            return None
    
    def predict_csv_file(self, model_name: str, csv_filepath: str) -> Optional[List[Dict]]:
        """Make predictions for matches in CSV file"""
        try:
            # Load CSV file
            df = pd.read_csv(csv_filepath)
            
            # Validate CSV format
            required_columns = ['HomeTeam', 'AwayTeam']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"CSV file must contain columns: {required_columns}")
                return None
            
            results = []
            
            # Make predictions for each match
            for _, row in df.iterrows():
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']
                
                prediction = self.predict_single_match(model_name, home_team, away_team)
                
                if prediction and 'error' not in prediction:
                    result = {
                        'HomeTeam': home_team,
                        'AwayTeam': away_team,
                        'Prediction': prediction['prediction'],
                        'Confidence': prediction['confidence'],
                        'HomeWinProb': prediction['probabilities']['home_win'],
                        'DrawProb': prediction['probabilities']['draw'],
                        'AwayWinProb': prediction['probabilities']['away_win']
                    }
                else:
                    result = {
                        'HomeTeam': home_team,
                        'AwayTeam': away_team,
                        'Prediction': 'Error',
                        'Confidence': 0,
                        'HomeWinProb': 0,
                        'DrawProb': 0,
                        'AwayWinProb': 0,
                        'Error': prediction.get('error', 'Unknown error') if prediction else 'Prediction failed'
                    }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting CSV file: {str(e)}")
            return None
    
    def predict_fixtures(self, model_name: str, fixtures: List[Dict]) -> Optional[List[Dict]]:
        """Make predictions for scraped fixtures"""
        try:
            results = []
            
            for fixture in fixtures:
                home_team = fixture.get('home_team')
                away_team = fixture.get('away_team')
                match_date = fixture.get('date')
                
                if not home_team or not away_team:
                    continue
                
                prediction = self.predict_single_match(model_name, home_team, away_team)
                
                if prediction and 'error' not in prediction:
                    result = {
                        'date': match_date,
                        'home_team': prediction['home_team'],
                        'away_team': prediction['away_team'],
                        'prediction': prediction['prediction'],
                        'confidence': prediction['confidence'],
                        'probabilities': prediction['probabilities']
                    }
                else:
                    result = {
                        'date': match_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'prediction': 'Error',
                        'confidence': 0,
                        'probabilities': {'home_win': 0, 'draw': 0, 'away_win': 0},
                        'error': prediction.get('error', 'Prediction failed') if prediction else 'Prediction failed'
                    }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting fixtures: {str(e)}")
            return None
    
    def analyze_model_performance(self, model_name: str) -> Optional[Dict]:
        """Generate comprehensive model performance analysis"""
        try:
            metadata = self.model_repo.get_model_metadata(model_name)
            if not metadata:
                return None
            
            analysis = {
                'model_info': {
                    'name': model_name,
                    'type': metadata['model_type'],
                    'league': metadata['league_name'],
                    'created_at': metadata['created_at'],
                    'accuracy': metadata.get('accuracy', 0)
                },
                'training_info': {
                    'training_samples': metadata.get('training_samples', 0),
                    'test_samples': metadata.get('test_samples', 0),
                    'features': metadata.get('features', 0),
                    'class_distribution': metadata.get('class_distribution', {})
                }
            }
            
            # Add detailed classification report if available
            if 'test_report' in metadata:
                analysis['performance_metrics'] = metadata['test_report']
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {str(e)}")
            return None
    
    def _save_scaler(self, model_name: str, scaler: StandardScaler):
        """Save scaler for a model"""
        try:
            import joblib
            scaler_path = f"database/storage/checkpoints/{model_name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            self.scalers[model_name] = scaler
        except Exception as e:
            logger.error(f"Error saving scaler for {model_name}: {str(e)}")
    
    def _load_scaler(self, model_name: str) -> Optional[StandardScaler]:
        """Load scaler for a model"""
        try:
            if model_name in self.scalers:
                return self.scalers[model_name]
            
            import joblib
            import os
            scaler_path = f"database/storage/checkpoints/{model_name}_scaler.pkl"
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                self.scalers[model_name] = scaler
                return scaler
            
            logger.error(f"Scaler not found for model {model_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading scaler for {model_name}: {str(e)}")
            return None
