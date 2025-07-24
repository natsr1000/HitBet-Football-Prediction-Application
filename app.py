"""
ProphitBet Football Prediction Web Application
A Flask-based web interface for football match outcome prediction using machine learning.
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from io import BytesIO
import zipfile

from config import LEAGUE_URLS, FIXTURE_URLS
from repositories.database_league_repository import DatabaseLeagueRepository
from repositories.database_model_repository import DatabaseModelRepository
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.scraping_service import ScrapingService
from utils.validators import validate_fixture_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prophitbet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'prophitbet-secret-key-2025')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs('database/storage/leagues/saved', exist_ok=True)
os.makedirs('database/storage/checkpoints', exist_ok=True)
os.makedirs('cache', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Initialize database and repositories
from database_config import get_db_manager

# Initialize database
db_manager = get_db_manager()

# Initialize services
league_repo = DatabaseLeagueRepository()
model_repo = DatabaseModelRepository()
data_service = DataService(league_repo)
prediction_service = PredictionService(model_repo, data_service)
scraping_service = ScrapingService()

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        # Get summary statistics
        saved_leagues = league_repo.get_saved_leagues()
        saved_models = model_repo.get_saved_models()
        
        stats = {
            'total_leagues': sum(len(leagues) for leagues in LEAGUE_URLS.values()),
            'saved_leagues': len(saved_leagues),
            'saved_models': len(saved_models),
            'countries': len(LEAGUE_URLS.keys())
        }
        
        return render_template('index.html', stats=stats, leagues=saved_leagues[:5])
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        return render_template('index.html', stats={}, leagues=[], error=str(e))

@app.route('/leagues')
def leagues():
    """League management page"""
    try:
        available_leagues = []
        for country, leagues in LEAGUE_URLS.items():
            for league_name, urls in leagues.items():
                available_leagues.append({
                    'country': country,
                    'name': league_name,
                    'seasons': len(urls)
                })
        
        saved_leagues = league_repo.get_saved_leagues()
        
        return render_template('leagues.html', 
                             available_leagues=available_leagues,
                             saved_leagues=saved_leagues)
    except Exception as e:
        logger.error(f"Error loading leagues page: {str(e)}")
        return render_template('leagues.html', 
                             available_leagues=[], 
                             saved_leagues=[], 
                             error=str(e))

@app.route('/api/leagues/download', methods=['POST'])
def download_league():
    """Download and save league data"""
    try:
        league_name = request.json.get('league_name')
        if not league_name:
            return jsonify({'error': 'League name is required'}), 400
        
        # Find league in LEAGUE_URLS
        league_found = False
        for country, leagues in LEAGUE_URLS.items():
            if league_name in leagues:
                league_found = True
                break
        
        if not league_found:
            return jsonify({'error': 'League not found'}), 404
        
        # Download league data
        success = data_service.download_league_data(league_name)
        
        if success:
            return jsonify({'message': f'Successfully downloaded {league_name} data'})
        else:
            return jsonify({'error': f'Failed to download {league_name} data'}), 500
            
    except Exception as e:
        logger.error(f"Error downloading league: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/leagues/delete', methods=['POST'])
def delete_league():
    """Delete saved league data"""
    try:
        league_name = request.json.get('league_name')
        if not league_name:
            return jsonify({'error': 'League name is required'}), 400
        
        success = league_repo.delete_league(league_name)
        
        if success:
            return jsonify({'message': f'Successfully deleted {league_name} data'})
        else:
            return jsonify({'error': f'Failed to delete {league_name} data'}), 500
            
    except Exception as e:
        logger.error(f"Error deleting league: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def models():
    """Model management page"""
    try:
        saved_models = model_repo.get_saved_models()
        saved_leagues = league_repo.get_saved_leagues()
        
        return render_template('models.html', 
                             saved_models=saved_models,
                             saved_leagues=saved_leagues)
    except Exception as e:
        logger.error(f"Error loading models page: {str(e)}")
        return render_template('models.html', 
                             saved_models=[], 
                             saved_leagues=[], 
                             error=str(e))

@app.route('/api/models/train', methods=['POST'])
def train_model():
    """Train a new model"""
    try:
        data = request.json
        league_name = data.get('league_name')
        model_type = data.get('model_type')
        model_name = data.get('model_name')
        
        if not all([league_name, model_type, model_name]):
            return jsonify({'error': 'League name, model type, and model name are required'}), 400
        
        if model_type not in ['fcnet', 'random_forest', 'simple_random_forest', 'simple_mlp']:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Check if league data exists
        if not league_repo.league_exists(league_name):
            return jsonify({'error': f'League data for {league_name} not found. Please download it first.'}), 404
        
        # Train model
        success, message = prediction_service.train_model(league_name, model_type, model_name)
        
        if success:
            return jsonify({'message': message})
        else:
            return jsonify({'error': message}), 500
            
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/delete', methods=['POST'])
def delete_model():
    """Delete a saved model"""
    try:
        model_name = request.json.get('model_name')
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        success = model_repo.delete_model(model_name)
        
        if success:
            return jsonify({'message': f'Successfully deleted model {model_name}'})
        else:
            return jsonify({'error': f'Failed to delete model {model_name}'}), 500
            
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predictions')
def predictions():
    """Predictions page"""
    try:
        saved_models = model_repo.get_saved_models()
        saved_leagues = league_repo.get_saved_leagues()
        
        # Get available leagues for fixture scraping
        fixture_leagues = list(FIXTURE_URLS.keys())
        
        return render_template('predictions.html', 
                             saved_models=saved_models,
                             saved_leagues=saved_leagues,
                             fixture_leagues=fixture_leagues)
    except Exception as e:
        logger.error(f"Error loading predictions page: {str(e)}")
        return render_template('predictions.html', 
                             saved_models=[], 
                             saved_leagues=[],
                             fixture_leagues=[], 
                             error=str(e))

@app.route('/api/predictions/single', methods=['POST'])
def predict_single():
    """Make prediction for a single match"""
    try:
        data = request.json
        model_name = data.get('model_name')
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not all([model_name, home_team, away_team]):
            return jsonify({'error': 'Model name, home team, and away team are required'}), 400
        
        # Make prediction
        result = prediction_service.predict_single_match(model_name, home_team, away_team)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        logger.error(f"Error making single prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/csv', methods=['POST'])
def predict_csv():
    """Make predictions for CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        model_name = request.form.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        try:
            # Make predictions
            results = prediction_service.predict_csv_file(model_name, filepath)
            
            if results:
                # Create results CSV
                results_df = pd.DataFrame(results)
                output = BytesIO()
                results_df.to_csv(output, index=False)
                output.seek(0)
                
                return send_file(
                    output,
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=f'predictions_{filename}'
                )
            else:
                return jsonify({'error': 'Prediction failed'}), 500
                
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
    except Exception as e:
        logger.error(f"Error making CSV predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/fixtures', methods=['POST'])
def predict_fixtures():
    """Make predictions for scraped fixtures"""
    try:
        data = request.json
        model_name = data.get('model_name')
        league_name = data.get('league_name')
        
        if not all([model_name, league_name]):
            return jsonify({'error': 'Model name and league name are required'}), 400
        
        if league_name not in FIXTURE_URLS:
            return jsonify({'error': 'League not supported for fixture scraping'}), 400
        
        # Scrape fixtures
        fixtures = scraping_service.scrape_fixtures(league_name)
        
        if not fixtures:
            return jsonify({'error': 'No fixtures found or scraping failed'}), 500
        
        # Make predictions
        results = prediction_service.predict_fixtures(model_name, fixtures)
        
        if results:
            return jsonify({'fixtures': results})
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        logger.error(f"Error predicting fixtures: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analysis')
def analysis():
    """Data analysis page"""
    try:
        saved_leagues = league_repo.get_saved_leagues()
        saved_models = model_repo.get_saved_models()
        
        return render_template('analysis.html', 
                             saved_leagues=saved_leagues,
                             saved_models=saved_models)
    except Exception as e:
        logger.error(f"Error loading analysis page: {str(e)}")
        return render_template('analysis.html', 
                             saved_leagues=[], 
                             saved_models=[],
                             error=str(e))

@app.route('/api/analysis/league', methods=['POST'])
def analyze_league():
    """Generate league analysis"""
    try:
        league_name = request.json.get('league_name')
        if not league_name:
            return jsonify({'error': 'League name is required'}), 400
        
        analysis = data_service.analyze_league_data(league_name)
        
        if analysis:
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Analysis failed'}), 500
            
    except Exception as e:
        logger.error(f"Error analyzing league: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/model', methods=['POST'])
def analyze_model():
    """Generate model analysis"""
    try:
        model_name = request.json.get('model_name')
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        analysis = prediction_service.analyze_model_performance(model_name)
        
        if analysis:
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Model analysis failed'}), 500
            
    except Exception as e:
        logger.error(f"Error analyzing model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)