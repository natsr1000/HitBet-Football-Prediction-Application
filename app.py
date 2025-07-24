"""
ProphitBet Football Prediction Web Application
A Flask-based web interface for football match outcome prediction using machine learning.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from io import BytesIO
import pandas as pd

# Configs and constants
from config import LEAGUE_URLS, FIXTURE_URLS

# Services and Repositories
from repositories.database_league_repository import DatabaseLeagueRepository
from repositories.database_model_repository import DatabaseModelRepository
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.scraping_service import ScrapingService
from utils.validators import validate_fixture_input

# Database setup
from database_config import db
from flask_migrate import Migrate

# App initialization
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'prophitbet-secret-key-2025')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

# Setup directories
os.makedirs('database/storage/leagues/saved', exist_ok=True)
os.makedirs('database/storage/checkpoints', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('cache', exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("prophitbet.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Repositories and Services
with app.app_context():
    league_repo = DatabaseLeagueRepository()
    model_repo = DatabaseModelRepository()
    data_service = DataService(league_repo)
    prediction_service = PredictionService(model_repo, data_service)
    scraping_service = ScrapingService()


@app.route('/')
def index():
    try:
        saved_leagues = league_repo.get_saved_leagues()
        saved_models = model_repo.get_saved_models()
        stats = {
            'total_leagues': sum(len(leagues) for leagues in LEAGUE_URLS.values()),
            'saved_leagues': len(saved_leagues),
            'saved_models': len(saved_models),
            'countries': len(LEAGUE_URLS)
        }
        return render_template('index.html', stats=stats, leagues=saved_leagues[:5])
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        return render_template('index.html', stats={}, leagues=[], error=str(e))


@app.route('/leagues')
def leagues():
    try:
        available_leagues = [
            {'country': country, 'name': name, 'seasons': len(urls)}
            for country, leagues in LEAGUE_URLS.items()
            for name, urls in leagues.items()
        ]
        saved_leagues = league_repo.get_saved_leagues()
        return render_template('leagues.html', available_leagues=available_leagues, saved_leagues=saved_leagues)
    except Exception as e:
        logger.error(f"Error loading leagues page: {str(e)}")
        return render_template('leagues.html', available_leagues=[], saved_leagues=[], error=str(e))


@app.route('/api/leagues/download', methods=['POST'])
def download_league():
    try:
        league_name = request.json.get('league_name')
        if not league_name:
            return jsonify({'error': 'League name is required'}), 400

        league_found = any(league_name in leagues for leagues in LEAGUE_URLS.values())
        if not league_found:
            return jsonify({'error': 'League not found'}), 404

        success = data_service.download_league_data(league_name)
        if success:
            return jsonify({'message': f'Successfully downloaded {league_name} data'})
        return jsonify({'error': f'Failed to download {league_name} data'}), 500

    except Exception as e:
        logger.error(f"Error downloading league: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/leagues/delete', methods=['POST'])
def delete_league():
    try:
        league_name = request.json.get('league_name')
        if not league_name:
            return jsonify({'error': 'League name is required'}), 400

        success = league_repo.delete_league(league_name)
        if success:
            return jsonify({'message': f'Deleted {league_name}'})
        return jsonify({'error': f'Failed to delete {league_name}'}), 500
    except Exception as e:
        logger.error(f"Error deleting league: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/models')
def models():
    try:
        saved_models = model_repo.get_saved_models()
        saved_leagues = league_repo.get_saved_leagues()
        return render_template('models.html', saved_models=saved_models, saved_leagues=saved_leagues)
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return render_template('models.html', saved_models=[], saved_leagues=[], error=str(e))


@app.route('/api/models/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        league_name = data.get('league_name')
        model_type = data.get('model_type')
        model_name = data.get('model_name')

        if not all([league_name, model_type, model_name]):
            return jsonify({'error': 'All fields required'}), 400

        if not league_repo.league_exists(league_name):
            return jsonify({'error': f'{league_name} data not found'}), 404

        success, message = prediction_service.train_model(league_name, model_type, model_name)
        if success:
            return jsonify({'message': message})
        return jsonify({'error': message}), 500
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/delete', methods=['POST'])
def delete_model():
    try:
        model_name = request.json.get('model_name')
        if not model_name:
            return jsonify({'error': 'Model name required'}), 400

        success = model_repo.delete_model(model_name)
        if success:
            return jsonify({'message': f'Deleted {model_name}'})
        return jsonify({'error': f'Failed to delete {model_name}'}), 500
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predictions')
def predictions():
    try:
        return render_template(
            'predictions.html',
            saved_models=model_repo.get_saved_models(),
            saved_leagues=league_repo.get_saved_leagues(),
            fixture_leagues=list(FIXTURE_URLS.keys())
        )
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        return render_template('predictions.html', error=str(e))


@app.route('/api/predictions/single', methods=['POST'])
def predict_single():
    try:
        data = request.json
        model_name = data.get('model_name')
        home_team = data.get('home_team')
        away_team = data.get('away_team')

        if not all([model_name, home_team, away_team]):
            return jsonify({'error': 'All fields required'}), 400

        result = prediction_service.predict_single_match(model_name, home_team, away_team)
        if result:
            return jsonify(result)
        return jsonify({'error': 'Prediction failed'}), 500
    except Exception as e:
        logger.error(f"Single prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        model_name = request.form.get('model_name')

        if not model_name or file.filename == '':
            return jsonify({'error': 'Missing fields or file'}), 400
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'CSV only'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        try:
            results = prediction_service.predict_csv_file(model_name, filepath)
            if results:
                output = BytesIO()
                pd.DataFrame(results).to_csv(output, index=False)
                output.seek(0)
                return send_file(output, mimetype='text/csv', as_attachment=True,
                                 download_name=f'predictions_{filename}')
            return jsonify({'error': 'Prediction failed'}), 500
        finally:
            os.remove(filepath)
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/fixtures', methods=['POST'])
def predict_fixtures():
    try:
        data = request.json
        model_name = data.get('model_name')
        league_name = data.get('league_name')

        if not all([model_name, league_name]):
            return jsonify({'error': 'All fields required'}), 400
        if league_name not in FIXTURE_URLS:
            return jsonify({'error': 'Unsupported league'}), 400

        fixtures = scraping_service.scrape_fixtures(league_name)
        if not fixtures:
            return jsonify({'error': 'No fixtures found'}), 500

        results = prediction_service.predict_fixtures(model_name, fixtures)
        return jsonify({'fixtures': results}) if results else jsonify({'error': 'Prediction failed'}), 500
    except Exception as e:
        logger.error(f"Fixture prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analysis')
def analysis():
    try:
        return render_template(
            'analysis.html',
            saved_leagues=league_repo.get_saved_leagues(),
            saved_models=model_repo.get_saved_models()
        )
    except Exception as e:
        logger.error(f"Error loading analysis: {str(e)}")
        return render_template('analysis.html', error=str(e))


@app.route('/api/analysis/league', methods=['POST'])
def analyze_league():
    try:
        league_name = request.json.get('league_name')
        if not league_name:
            return jsonify({'error': 'League required'}), 400

        analysis = data_service.analyze_league_data(league_name)
        return jsonify(analysis) if analysis else jsonify({'error': 'Analysis failed'}), 500
    except Exception as e:
        logger.error(f"League analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/model', methods=['POST'])
def analyze_model():
    try:
        model_name = request.json.get('model_name')
        if not model_name:
            return jsonify({'error': 'Model required'}), 400

        analysis = prediction_service.analyze_model_performance(model_name)
        return jsonify(analysis) if analysis else jsonify({'error': 'Analysis failed'}), 500
    except Exception as e:
        logger.error(f"Model analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Internal server error"), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
