"""
Database-based repository for ML model management
"""

import logging
import pickle
import base64
import json
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from datetime import datetime

from models.database_models import MLModel, League
from database_config import get_db_session, close_db_session

logger = logging.getLogger(__name__)

class DatabaseModelRepository:
    """Database-based repository for ML model management"""
    
    def save_model(self, model_name: str, model: Any, model_type: str, 
                   league_name: str, metadata: Dict) -> bool:
        """Save a trained model with metadata to database"""
        session = get_db_session()
        try:
            # Get league ID
            league = session.query(League).filter(League.name == league_name).first()
            if not league:
                logger.error(f"League {league_name} not found")
                return False
            
            # Serialize model
            try:
                model_bytes = pickle.dumps(model)
                model_data = base64.b64encode(model_bytes).decode('utf-8')
            except Exception as e:
                logger.error(f"Error serializing model: {str(e)}")
                return False
            
            # Convert numpy types in metadata for JSON storage
            def convert_numpy_types(obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj
            
            clean_metadata = convert_numpy_types(metadata)
            
            # Check if model already exists
            existing_model = session.query(MLModel).filter(MLModel.name == model_name).first()
            
            if existing_model:
                # Update existing model
                existing_model.model_type = model_type
                existing_model.league_id = league.id
                existing_model.accuracy = clean_metadata.get('accuracy')
                existing_model.training_samples = clean_metadata.get('training_samples')
                existing_model.test_samples = clean_metadata.get('test_samples')
                existing_model.features = clean_metadata.get('features')
                existing_model.class_distribution = clean_metadata.get('class_distribution')
                existing_model.test_report = clean_metadata.get('test_report')
                existing_model.model_data = model_data
                existing_model.updated_at = datetime.utcnow()
                existing_model.is_active = True
            else:
                # Create new model
                ml_model = MLModel(
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
                session.add(ml_model)
            
            session.commit()
            logger.info(f"Saved model {model_name} to database")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving model {model_name}: {str(e)}")
            return False
        finally:
            close_db_session(session)
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a trained model from database"""
        session = get_db_session()
        try:
            ml_model = session.query(MLModel).filter(
                MLModel.name == model_name,
                MLModel.is_active == True
            ).first()
            
            if not ml_model:
                logger.error(f"Model {model_name} not found in database")
                return None
            
            # Deserialize model
            try:
                model_bytes = base64.b64decode(ml_model.model_data.encode('utf-8'))
                model = pickle.loads(model_bytes)
                logger.info(f"Loaded model {model_name} from database")
                return model
            except Exception as e:
                logger.error(f"Error deserializing model {model_name}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
        finally:
            close_db_session(session)
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a specific model"""
        session = get_db_session()
        try:
            ml_model = session.query(MLModel).filter(
                MLModel.name == model_name,
                MLModel.is_active == True
            ).first()
            
            if not ml_model:
                return None
            
            # Get league name
            league = session.query(League).filter(League.id == ml_model.league_id).first()
            league_name = league.name if league else "Unknown"
            
            metadata = {
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
                'updated_at': ml_model.updated_at.isoformat()
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting model metadata for {model_name}: {str(e)}")
            return None
        finally:
            close_db_session(session)
    
    def get_saved_models(self) -> List[Dict]:
        """Get list of all saved models"""
        session = get_db_session()
        try:
            models = session.query(MLModel).filter(MLModel.is_active == True).all()
            
            result = []
            for model in models:
                # Get league name
                league = session.query(League).filter(League.id == model.league_id).first()
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
            
        except Exception as e:
            logger.error(f"Error getting saved models: {str(e)}")
            return []
        finally:
            close_db_session(session)
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a saved model"""
        session = get_db_session()
        try:
            ml_model = session.query(MLModel).filter(MLModel.name == model_name).first()
            
            if not ml_model:
                return False
            
            # Soft delete by setting is_active to False
            ml_model.is_active = False
            ml_model.updated_at = datetime.utcnow()
            
            session.commit()
            logger.info(f"Deleted model {model_name}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting model {model_name}: {str(e)}")
            return False
        finally:
            close_db_session(session)
    
    def save_scaler(self, model_name: str, scaler: Any) -> bool:
        """Save scaler data for a model"""
        session = get_db_session()
        try:
            ml_model = session.query(MLModel).filter(
                MLModel.name == model_name,
                MLModel.is_active == True
            ).first()
            
            if not ml_model:
                logger.error(f"Model {model_name} not found")
                return False
            
            # Serialize scaler
            try:
                scaler_bytes = pickle.dumps(scaler)
                scaler_data = base64.b64encode(scaler_bytes).decode('utf-8')
                
                ml_model.scaler_data = scaler_data
                ml_model.updated_at = datetime.utcnow()
                
                session.commit()
                logger.info(f"Saved scaler for model {model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error serializing scaler: {str(e)}")
                return False
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving scaler for {model_name}: {str(e)}")
            return False
        finally:
            close_db_session(session)
    
    def load_scaler(self, model_name: str) -> Optional[Any]:
        """Load scaler for a model"""
        session = get_db_session()
        try:
            ml_model = session.query(MLModel).filter(
                MLModel.name == model_name,
                MLModel.is_active == True
            ).first()
            
            if not ml_model or not ml_model.scaler_data:
                return None
            
            # Deserialize scaler
            try:
                scaler_bytes = base64.b64decode(ml_model.scaler_data.encode('utf-8'))
                scaler = pickle.loads(scaler_bytes)
                return scaler
            except Exception as e:
                logger.error(f"Error deserializing scaler for {model_name}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading scaler for {model_name}: {str(e)}")
            return None
        finally:
            close_db_session(session)