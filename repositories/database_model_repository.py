"""
Database-based repository for ML model management
"""

import logging
import pickle
import base64
from typing import List, Dict, Optional, Any
from datetime import datetime

from models.database_models import MLModel, League
from database_config import db_session

logger = logging.getLogger(__name__)

class DatabaseModelRepository:
    """Database-based repository for ML model management"""

    def save_model(self, model_name: str, model: Any, model_type: str,
                   league_name: str, metadata: Dict) -> bool:
        """Save a trained model with metadata to database"""
        with db_session() as session:
            league = session.query(League).filter_by(name=league_name).first()
            if not league:
                logger.error(f"League {league_name} not found")
                return False

            try:
                model_bytes = pickle.dumps(model)
                model_data = base64.b64encode(model_bytes).decode('utf-8')
            except Exception as e:
                logger.error(f"Error serializing model: {str(e)}")
                return False

            def convert(obj):
                import numpy as np
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list): return [convert(v) for v in obj]
                return obj

            clean_metadata = convert(metadata)

            model_obj = session.query(MLModel).filter_by(name=model_name).first()
            if model_obj:
                # Update
                model_obj.model_type = model_type
                model_obj.league_id = league.id
                model_obj.accuracy = clean_metadata.get('accuracy')
                model_obj.training_samples = clean_metadata.get('training_samples')
                model_obj.test_samples = clean_metadata.get('test_samples')
                model_obj.features = clean_metadata.get('features')
                model_obj.class_distribution = clean_metadata.get('class_distribution')
                model_obj.test_report = clean_metadata.get('test_report')
                model_obj.model_data = model_data
                model_obj.updated_at = datetime.utcnow()
                model_obj.is_active = True
            else:
                # Insert
                new_model = MLModel(
                    name=model_name,
                    model_type=model_type,
                    league_id=league.id,
                    accuracy=clean_metadata.get('accuracy'),
                    training_samples=clean_metadata.get('training_samples'),
                    test_samples=clean_metadata.get('test_samples'),
                    features=clean_metadata.get('features'),
                    class_distribution=clean_metadata.get('class_distribution'),
                    test_report=clean_metadata.get('test_report'),
                    model_data=model_data,
                    is_active=True
                )
                session.add(new_model)

            logger.info(f"Saved model {model_name} to database")
            return True

    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a trained model from database"""
        with db_session() as session:
            ml_model = session.query(MLModel).filter_by(name=model_name, is_active=True).first()
            if not ml_model:
                logger.error(f"Model {model_name} not found")
                return None
            try:
                model_bytes = base64.b64decode(ml_model.model_data.encode('utf-8'))
                return pickle.loads(model_bytes)
            except Exception as e:
                logger.error(f"Error deserializing model {model_name}: {str(e)}")
                return None

    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a specific model"""
        with db_session() as session:
            ml_model = session.query(MLModel).filter_by(name=model_name, is_active=True).first()
            if not ml_model:
                return None
            league = session.query(League).filter_by(id=ml_model.league_id).first()
            league_name = league.name if league else "Unknown"
            return {
                'model_name': ml_model.name,
                'model_type': ml_model.model_type,
                'league_name': league_name,
                'accuracy': ml_model.accuracy,
                'training_samples': ml_model.training_samples,
                'test_samples': ml_model.test_samples,
                'features': ml_model.features,
                'class_distribution': ml_model.class_distribution,
                'test_report': ml_model.test_report,
                'created_at': ml_model.created_at.isoformat(),
                'updated_at': ml_model.updated_at.isoformat(),
            }

    def get_saved_models(self) -> List[Dict]:
        """Get list of all saved models"""
        with db_session() as session:
            models = session.query(MLModel).filter_by(is_active=True).all()
            result = []
            for model in models:
                league = session.query(League).filter_by(id=model.league_id).first()
                league_name = league.name if league else "Unknown"
                result.append({
                    'model_name': model.name,
                    'model_type': model.model_type,
                    'league_name': league_name,
                    'accuracy': model.accuracy,
                    'training_samples': model.training_samples,
                    'test_samples': model.test_samples,
                    'created_at': model.created_at.isoformat(),
                    'updated_at': model.updated_at.isoformat()
                })
            return sorted(result, key=lambda x: x['created_at'], reverse=True)

    def delete_model(self, model_name: str) -> bool:
        """Soft delete a saved model"""
        with db_session() as session:
            ml_model = session.query(MLModel).filter_by(name=model_name).first()
            if not ml_model:
                return False
            ml_model.is_active = False
            ml_model.updated_at = datetime.utcnow()
            logger.info(f"Deleted model {model_name}")
            return True

    def save_scaler(self, model_name: str, scaler: Any) -> bool:
        """Save scaler data for a model"""
        with db_session() as session:
            ml_model = session.query(MLModel).filter_by(name=model_name, is_active=True).first()
            if not ml_model:
                logger.error(f"Model {model_name} not found")
                return False
            try:
                scaler_bytes = pickle.dumps(scaler)
                scaler_data = base64.b64encode(scaler_bytes).decode('utf-8')
                ml_model.scaler_data = scaler_data
                ml_model.updated_at = datetime.utcnow()
                logger.info(f"Saved scaler for model {model_name}")
                return True
            except Exception as e:
                logger.error(f"Error serializing scaler: {str(e)}")
                return False

    def load_scaler(self, model_name: str) -> Optional[Any]:
        """Load scaler for a model"""
        with db_session() as session:
            ml_model = session.query(MLModel).filter_by(name=model_name, is_active=True).first()
            if not ml_model or not ml_model.scaler_data:
                return None
            try:
                scaler_bytes = base64.b64decode(ml_model.scaler_data.encode('utf-8'))
                return pickle.loads(scaler_bytes)
            except Exception as e:
                logger.error(f"Error deserializing scaler for {model_name}: {str(e)}")
                return None
