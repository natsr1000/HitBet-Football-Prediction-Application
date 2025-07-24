"""
Database-based repository for ML model management
"""

import logging
import pickle
import base64
from typing import List, Dict, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session

from models.database_models import MLModel, League
from database_config import db_session

logger = logging.getLogger(__name__)

class DatabaseModelRepository:
    """Database-based repository for ML model management"""

    def save_model(self, model_name: str, model: Any, model_type: str,
                   league_name: str, metadata: Dict) -> bool:
        """Save a trained model with metadata to database"""
        try:
            with db_session() as session:
                league = session.query(League).filter_by(name=league_name).first()
                if not league:
                    logger.error(f"League '{league_name}' not found.")
                    return False

                # Serialize model
                model_bytes = pickle.dumps(model)
                model_data = base64.b64encode(model_bytes).decode('utf-8')

                # Clean metadata (convert numpy types)
                def convert(obj):
                    import numpy as np
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert(v) for v in obj]
                    return obj

                clean_meta = convert(metadata)

                existing = session.query(MLModel).filter_by(name=model_name).first()
                if existing:
                    # Update
                    existing.model_type = model_type
                    existing.league_id = league.id
                    existing.accuracy = clean_meta.get("accuracy")
                    existing.training_samples = clean_meta.get("training_samples")
                    existing.test_samples = clean_meta.get("test_samples")
                    existing.features = clean_meta.get("features")
                    existing.class_distribution = clean_meta.get("class_distribution")
                    existing.test_report = clean_meta.get("test_report")
                    existing.model_data = model_data
                    existing.updated_at = datetime.utcnow()
                    existing.is_active = True
                else:
                    new_model = MLModel(
                        name=model_name,
                        model_type=model_type,
                        league_id=league.id,
                        accuracy=clean_meta.get("accuracy"),
                        training_samples=clean_meta.get("training_samples"),
                        test_samples=clean_meta.get("test_samples"),
                        features=clean_meta.get("features"),
                        class_distribution=clean_meta.get("class_distribution"),
                        test_report=clean_meta.get("test_report"),
                        model_data=model_data,
                        is_active=True
                    )
                    session.add(new_model)
                logger.info(f"Saved model '{model_name}' to database.")
                return True
        except Exception as e:
            logger.error(f"Error saving model '{model_name}': {str(e)}")
            return False

    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a trained model from the database"""
        try:
            with db_session() as session:
                model = session.query(MLModel).filter_by(name=model_name, is_active=True).first()
                if not model:
                    logger.error(f"Model '{model_name}' not found.")
                    return None

                model_bytes = base64.b64decode(model.model_data.encode('utf-8'))
                return pickle.loads(model_bytes)
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {str(e)}")
            return None

    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Retrieve metadata for a specific model"""
        try:
            with db_session() as session:
                model = session.query(MLModel).filter_by(name=model_name, is_active=True).first()
                if not model:
                    return None

                league = session.query(League).get(model.league_id)
                return {
                    'model_name': model.name,
                    'model_type': model.model_type,
                    'league_name': league.name if league else "Unknown",
                    'accuracy': model.accuracy,
                    'training_samples': model.training_samples,
                    'test_samples': model.test_samples,
                    'features': model.features,
                    'class_distribution': model.class_distribution,
                    'test_report': model.test_report,
                    'created_at': model.created_at.isoformat(),
                    'updated_at': model.updated_at.isoformat()
                }
        except Exception as e:
            logger.error(f"Error retrieving metadata for model '{model_name}': {str(e)}")
            return None

    def get_saved_models(self) -> List[Dict]:
        """Retrieve list of all saved models"""
        try:
            with db_session() as session:
                models = session.query(MLModel).filter_by(is_active=True).all()
                results = []
                for m in models:
                    league = session.query(League).get(m.league_id)
                    results.append({
                        'model_name': m.name,
                        'model_type': m.model_type,
                        'league_name': league.name if league else "Unknown",
                        'accuracy': m.accuracy,
                        'training_samples': m.training_samples,
                        'test_samples': m.test_samples,
                        'created_at': m.created_at.isoformat(),
                        'updated_at': m.updated_at.isoformat()
                    })
                return sorted(results, key=lambda x: x['created_at'], reverse=True)
        except Exception as e:
            logger.error(f"Error retrieving saved models: {str(e)}")
            return []

    def delete_model(self, model_name: str) -> bool:
        """Soft delete a model by setting is_active=False"""
        try:
            with db_session() as session:
                model = session.query(MLModel).filter_by(name=model_name).first()
                if not model:
                    return False
                model.is_active = False
                model.updated_at = datetime.utcnow()
                logger.info(f"Model '{model_name}' deleted (soft).")
                return True
        except Exception as e:
            logger.error(f"Error deleting model '{model_name}': {str(e)}")
            return False

    def save_scaler(self, model_name: str, scaler: Any) -> bool:
        """Save scaler object for a given model"""
        try:
            with db_session() as session:
                model = session.query(MLModel).filter_by(name=model_name, is_active=True).first()
                if not model:
                    logger.error(f"Model '{model_name}' not found.")
                    return False
                scaler_bytes = pickle.dumps(scaler)
                model.scaler_data = base64.b64encode(scaler_bytes).decode('utf-8')
                model.updated_at = datetime.utcnow()
                logger.info(f"Scaler for model '{model_name}' saved.")
                return True
        except Exception as e:
            logger.error(f"Error saving scaler for '{model_name}': {str(e)}")
            return False

    def load_scaler(self, model_name: str) -> Optional[Any]:
        """Load scaler for a given model"""
        try:
            with db_session() as session:
                model = session.query(MLModel).filter_by(name=model_name, is_active=True).first()
                if not model or not model.scaler_data:
                    return None
                scaler_bytes = base64.b64decode(model.scaler_data.encode('utf-8'))
                return pickle.loads(scaler_bytes)
        except Exception as e:
            logger.error(f"Error loading scaler for '{model_name}': {str(e)}")
            return None
