"""
Repository for managing trained model storage and retrieval
"""

import os
import json
import joblib
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class ModelRepository:
    """Handles persistence and retrieval of trained models"""
    
    def __init__(self):
        self.models_dir = 'database/storage/checkpoints/'
        self.metadata_file = 'model_metadata.json'
        
        # Ensure directory exists
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model(self, model_name: str, model: Any, model_type: str, 
                   league_name: str, metadata: Dict) -> bool:
        """Save a trained model with metadata"""
        try:
            # Save model file
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            
            if model_type == 'fcnet':
                # For Keras models, save as .h5
                model_path = os.path.join(self.models_dir, f"{model_name}.h5")
                model.save(model_path)
            else:
                # For scikit-learn models, use joblib
                joblib.dump(model, model_path)
            
            # Save metadata
            model_metadata = {
                'model_name': model_name,
                'model_type': model_type,
                'league_name': league_name,
                'created_at': datetime.now().isoformat(),
                'file_path': model_path,
                **metadata
            }
            
            self._save_metadata(model_name, model_metadata)
            logger.info(f"Saved model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            return False
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a trained model"""
        try:
            metadata = self._load_metadata(model_name)
            if not metadata:
                return None
            
            model_path = metadata['file_path']
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            if metadata['model_type'] == 'fcnet':
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
            else:
                model = joblib.load(model_path)
            
            logger.info(f"Loaded model {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def model_exists(self, model_name: str) -> bool:
        """Check if model exists"""
        metadata = self._load_metadata(model_name)
        if not metadata:
            return False
        
        return os.path.exists(metadata['file_path'])
    
    def get_saved_models(self) -> List[Dict]:
        """Get list of saved models with metadata"""
        models = []
        try:
            metadata_path = os.path.join(self.models_dir, self.metadata_file)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    all_metadata = json.load(f)
                
                for model_name, metadata in all_metadata.items():
                    if os.path.exists(metadata['file_path']):
                        models.append(metadata)
            
        except Exception as e:
            logger.error(f"Error getting saved models: {str(e)}")
        
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a saved model"""
        try:
            metadata = self._load_metadata(model_name)
            if not metadata:
                return False
            
            # Delete model file
            model_path = metadata['file_path']
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Remove from metadata
            self._remove_metadata(model_name)
            
            logger.info(f"Deleted model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {str(e)}")
            return False
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a specific model"""
        return self._load_metadata(model_name)
    
    def _save_metadata(self, model_name: str, metadata: Dict):
        """Save model metadata to JSON file"""
        try:
            # Convert numpy types to Python types for JSON serialization
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
            
            converted_metadata = convert_numpy_types(metadata)
            
            metadata_path = os.path.join(self.models_dir, self.metadata_file)
            
            # Load existing metadata
            all_metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        all_metadata = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted metadata file, creating new one")
                    all_metadata = {}
            
            # Add new metadata
            all_metadata[model_name] = converted_metadata
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(all_metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metadata for {model_name}: {str(e)}")
    
    def _load_metadata(self, model_name: str) -> Optional[Dict]:
        """Load metadata for a specific model"""
        try:
            metadata_path = os.path.join(self.models_dir, self.metadata_file)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    all_metadata = json.load(f)
                return all_metadata.get(model_name)
        except Exception as e:
            logger.error(f"Error loading metadata for {model_name}: {str(e)}")
        
        return None
    
    def _remove_metadata(self, model_name: str):
        """Remove metadata for a specific model"""
        try:
            metadata_path = os.path.join(self.models_dir, self.metadata_file)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    all_metadata = json.load(f)
                
                if model_name in all_metadata:
                    del all_metadata[model_name]
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(all_metadata, f, indent=2)
                        
        except Exception as e:
            logger.error(f"Error removing metadata for {model_name}: {str(e)}")