"""
Service for data management including downloading, processing, and analysis
"""

import os
import pandas as pd
import numpy as np
import requests
import logging
from typing import List, Dict, Optional, Tuple
# from tqdm import tqdm  # Commented out to avoid dependency issues
import matplotlib.pyplot as plt
# import seaborn as sns  # Commented out to avoid dependency issues
from io import BytesIO
import base64

from config import LEAGUE_URLS
from repositories.league_repository import LeagueRepository
from utils.feature_engineering import compute_team_statistics

logger = logging.getLogger(__name__)

class DataService:
    """Service for managing football data operations"""
    
    def __init__(self, league_repo: LeagueRepository):
        self.league_repo = league_repo
    
    def download_league_data(self, league_name: str) -> bool:
        """Download and process league data from football-data.co.uk"""
        try:
            # Special handling for sample data
            if league_name == "England-Premier League":
                return self._load_sample_data(league_name)
            
            # Find league URLs
            league_urls = None
            for country, leagues in LEAGUE_URLS.items():
                if league_name in leagues:
                    league_urls = leagues[league_name]
                    break
            
            if not league_urls:
                logger.error(f"League {league_name} not found in available leagues")
                return False
            
            all_data = []
            
            # Download data for each season
            logger.info(f"Downloading {league_name} data from {len(league_urls)} seasons...")
            for i, (url, season) in enumerate(league_urls):
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Read CSV data
                    data = pd.read_csv(BytesIO(response.content))
                    
                    # Add season column
                    data['Season'] = season
                    
                    # Basic data cleaning
                    data = self._clean_data(data)
                    
                    if len(data) > 0:
                        all_data.append(data)
                        logger.info(f"Downloaded {len(data)} matches for {league_name} season {season}")
                    
                except Exception as e:
                    logger.warning(f"Failed to download {league_name} season {season}: {str(e)}")
                    continue
            
            if not all_data:
                logger.error(f"No data downloaded for {league_name}")
                return False
            
            # Combine all seasons
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Additional data processing
            processed_data = self._process_data(combined_data)
            
            # Save to repository
            success = self.league_repo.save_league_data(league_name, processed_data)
            
            if success:
                logger.info(f"Successfully downloaded and saved {len(processed_data)} matches for {league_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error downloading league data for {league_name}: {str(e)}")
            return False
    
    def _load_sample_data(self, league_name: str) -> bool:
        """Load sample data for demonstration"""
        try:
            import os
            sample_file = "sample_data/premier_league_sample.csv"
            
            if not os.path.exists(sample_file):
                logger.error(f"Sample data file not found: {sample_file}")
                return False
            
            # Read sample data
            data = pd.read_csv(sample_file)
            
            # Add season column
            data['Season'] = 2024
            
            # Clean and process data
            cleaned_data = self._clean_data(data)
            processed_data = self._process_data(cleaned_data)
            
            if len(processed_data) == 0:
                logger.error("No valid data after processing sample data")
                return False
            
            # Save to repository
            success = self.league_repo.save_league_data(league_name, processed_data)
            
            if success:
                logger.info(f"Successfully loaded {len(processed_data)} sample matches for {league_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading sample data: {str(e)}")
            return False
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        try:
            # Required columns for predictions
            required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            
            # Check if required columns exist
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            # Remove rows with missing values in critical columns
            data = data.dropna(subset=required_cols)
            
            # Validate score data
            data = data[(data['FTHG'] >= 0) & (data['FTAG'] >= 0)]
            
            # Validate result data
            data = data[data['FTR'].isin(['H', 'D', 'A'])]
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return pd.DataFrame()
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and enhance data with additional features"""
        try:
            # Sort by date if available
            if 'Date' in data.columns:
                try:
                    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                    data = data.sort_values('Date')
                except:
                    pass
            
            # Add goal difference
            data['GoalDiff'] = data['FTHG'] - data['FTAG']
            
            # Add total goals
            data['TotalGoals'] = data['FTHG'] + data['FTAG']
            
            # Convert result to numeric for easier processing
            data['Result_Numeric'] = data['FTR'].map({'H': 1, 'D': 0, 'A': -1})
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return data
    
    def analyze_league_data(self, league_name: str) -> Optional[Dict]:
        """Generate comprehensive analysis of league data"""
        try:
            data = self.league_repo.load_league_data(league_name)
            if data is None:
                return None
            
            analysis = {}
            
            # Basic statistics
            analysis['basic_stats'] = {
                'total_matches': len(data),
                'total_goals': int(data['FTHG'].sum() + data['FTAG'].sum()),
                'avg_goals_per_match': round((data['FTHG'].sum() + data['FTAG'].sum()) / len(data), 2),
                'seasons': data['Season'].nunique() if 'Season' in data.columns else 1
            }
            
            # Outcome distribution
            outcome_counts = data['FTR'].value_counts()
            analysis['outcome_distribution'] = {
                'home_wins': int(outcome_counts.get('H', 0)),
                'draws': int(outcome_counts.get('D', 0)),
                'away_wins': int(outcome_counts.get('A', 0))
            }
            
            # Goal statistics
            analysis['goal_stats'] = {
                'avg_home_goals': round(data['FTHG'].mean(), 2),
                'avg_away_goals': round(data['FTAG'].mean(), 2),
                'highest_scoring_match': int(data['TotalGoals'].max()),
                'most_common_score': self._get_most_common_score(data)
            }
            
            # Team analysis
            teams = set(data['HomeTeam'].unique()) | set(data['AwayTeam'].unique())
            analysis['team_stats'] = {
                'num_teams': len(teams),
                'teams': sorted(list(teams))
            }
            
            # Generate charts
            charts = self._generate_analysis_charts(data)
            analysis.update(charts)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing league data for {league_name}: {str(e)}")
            return None
    
    def _get_most_common_score(self, data: pd.DataFrame) -> str:
        """Find the most common final score"""
        try:
            score_counts = data.groupby(['FTHG', 'FTAG']).size()
            most_common = score_counts.idxmax()
            return f"{most_common[0]}-{most_common[1]}"
        except:
            return "N/A"
    
    def _generate_analysis_charts(self, data: pd.DataFrame) -> Dict:
        """Generate analysis charts as base64 encoded images"""
        charts = {}
        
        try:
            # Outcome distribution pie chart
            plt.figure(figsize=(8, 6))
            outcome_counts = data['FTR'].value_counts()
            labels = ['Home Wins', 'Draws', 'Away Wins']
            colors = ['#2E8B57', '#FFD700', '#DC143C']
            
            plt.pie(outcome_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
            plt.title('Match Outcome Distribution')
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            charts['outcome_chart'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Goals distribution histogram
            plt.figure(figsize=(10, 6))
            plt.hist(data['TotalGoals'], bins=range(0, int(data['TotalGoals'].max()) + 2), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Total Goals per Match')
            plt.ylabel('Frequency')
            plt.title('Goals per Match Distribution')
            plt.grid(True, alpha=0.3)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            charts['goals_chart'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Season comparison if available
            if 'Season' in data.columns and data['Season'].nunique() > 1:
                plt.figure(figsize=(12, 6))
                season_goals = data.groupby('Season')['TotalGoals'].mean()
                
                plt.plot(season_goals.index, season_goals.values, marker='o', linewidth=2, markersize=6)
                plt.xlabel('Season')
                plt.ylabel('Average Goals per Match')
                plt.title('Average Goals per Match by Season')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                charts['season_chart'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
        
        return charts
    
    def prepare_training_data(self, league_name: str, num_recent_matches: int = 10) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Prepare data for model training"""
        try:
            data = self.league_repo.load_league_data(league_name)
            if data is None:
                return None
            
            # Sort by date if available
            if 'Date' in data.columns:
                try:
                    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                    data = data.sort_values('Date')
                except:
                    pass
            
            features = []
            labels = []
            
            # Get unique teams
            teams = sorted(set(data['HomeTeam'].unique()) | set(data['AwayTeam'].unique()))
            
            # Calculate team statistics for each match
            for idx, match in data.iterrows():
                home_team = match['HomeTeam']
                away_team = match['AwayTeam']
                
                # Get historical data up to this match
                historical_data = data.iloc[:idx]
                
                if len(historical_data) < 5:  # Need minimum historical data
                    continue
                
                # Compute team statistics
                home_stats = compute_team_statistics(historical_data, home_team, num_recent_matches)
                away_stats = compute_team_statistics(historical_data, away_team, num_recent_matches)
                
                if home_stats is not None and away_stats is not None:
                    # Combine features
                    match_features = np.concatenate([home_stats, away_stats])
                    features.append(match_features)
                    
                    # Add label
                    result = match['FTR']
                    if result == 'H':
                        labels.append(0)  # Home win
                    elif result == 'D':
                        labels.append(1)  # Draw
                    else:
                        labels.append(2)  # Away win
            
            if len(features) == 0:
                logger.error(f"No features generated for {league_name}")
                return None
            
            X = np.array(features)
            y = np.array(labels)
            
            logger.info(f"Prepared training data for {league_name}: {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data for {league_name}: {str(e)}")
            return None