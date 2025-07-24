"""
Repository for managing league data storage and retrieval
"""

import os
import pandas as pd
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class LeagueRepository:
    """Handles persistence and retrieval of league data"""
    
    def __init__(self):
        self.saved_leagues_dir = 'database/storage/leagues/saved/'
        self.available_leagues_file = 'database/storage/leagues/available_leagues.csv'
        
        # Ensure directories exist
        os.makedirs(self.saved_leagues_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.available_leagues_file), exist_ok=True)
    
    def save_league_data(self, league_name: str, data: pd.DataFrame) -> bool:
        """Save league data to CSV file"""
        try:
            filepath = os.path.join(self.saved_leagues_dir, f"{league_name.replace(' ', '_')}.csv")
            data.to_csv(filepath, index=False)
            logger.info(f"Saved league data for {league_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving league data for {league_name}: {str(e)}")
            return False
    
    def load_league_data(self, league_name: str) -> Optional[pd.DataFrame]:
        """Load league data from CSV file"""
        try:
            filepath = os.path.join(self.saved_leagues_dir, f"{league_name.replace(' ', '_')}.csv")
            if os.path.exists(filepath):
                data = pd.read_csv(filepath)
                logger.info(f"Loaded league data for {league_name} with {len(data)} matches")
                return data
            else:
                logger.warning(f"League data file not found for {league_name}")
                return None
        except Exception as e:
            logger.error(f"Error loading league data for {league_name}: {str(e)}")
            return None
    
    def league_exists(self, league_name: str) -> bool:
        """Check if league data exists"""
        filepath = os.path.join(self.saved_leagues_dir, f"{league_name.replace(' ', '_')}.csv")
        return os.path.exists(filepath)
    
    def get_saved_leagues(self) -> List[Dict]:
        """Get list of saved leagues with metadata"""
        leagues = []
        try:
            if os.path.exists(self.saved_leagues_dir):
                for filename in os.listdir(self.saved_leagues_dir):
                    if filename.endswith('.csv'):
                        league_name = filename.replace('_', ' ').replace('.csv', '')
                        filepath = os.path.join(self.saved_leagues_dir, filename)
                        
                        # Get file stats
                        stat = os.stat(filepath)
                        size_mb = round(stat.st_size / (1024 * 1024), 2)
                        
                        # Get number of matches
                        try:
                            df = pd.read_csv(filepath)
                            num_matches = len(df)
                        except:
                            num_matches = 0
                        
                        leagues.append({
                            'name': league_name,
                            'filename': filename,
                            'size_mb': size_mb,
                            'num_matches': num_matches,
                            'last_modified': stat.st_mtime
                        })
        except Exception as e:
            logger.error(f"Error getting saved leagues: {str(e)}")
        
        return sorted(leagues, key=lambda x: x['last_modified'], reverse=True)
    
    def get_available_leagues(self) -> List[Dict]:
        """Get list of available leagues for download"""
        leagues = []
        try:
            from config import LEAGUE_URLS
            for country, country_leagues in LEAGUE_URLS.items():
                for league_name, urls in country_leagues.items():
                    leagues.append({
                        'country': country,
                        'name': league_name,
                        'seasons': len(urls),
                        'is_saved': self.league_exists(league_name)
                    })
        except Exception as e:
            logger.error(f"Error getting available leagues: {str(e)}")
        
        return leagues
    
    def delete_league(self, league_name: str) -> bool:
        """Delete saved league data"""
        try:
            filepath = os.path.join(self.saved_leagues_dir, f"{league_name.replace(' ', '_')}.csv")
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted league data for {league_name}")
                return True
            else:
                logger.warning(f"League data file not found for {league_name}")
                return False
        except Exception as e:
            logger.error(f"Error deleting league data for {league_name}: {str(e)}")
            return False
    
    def get_league_statistics(self, league_name: str) -> Optional[Dict]:
        """Get statistics for a league"""
        try:
            data = self.load_league_data(league_name)
            if data is None:
                return None
            
            # Calculate basic statistics
            total_matches = len(data)
            seasons = data['Season'].nunique() if 'Season' in data.columns else 1
            teams = set()
            
            if 'HomeTeam' in data.columns and 'AwayTeam' in data.columns:
                teams.update(data['HomeTeam'].unique())
                teams.update(data['AwayTeam'].unique())
            
            # Calculate outcome distribution
            outcomes = {'Home': 0, 'Draw': 0, 'Away': 0}
            if 'FTR' in data.columns:
                outcomes['Home'] = (data['FTR'] == 'H').sum()
                outcomes['Draw'] = (data['FTR'] == 'D').sum()
                outcomes['Away'] = (data['FTR'] == 'A').sum()
            
            return {
                'total_matches': total_matches,
                'seasons': seasons,
                'num_teams': len(teams),
                'outcomes': outcomes,
                'teams': sorted(list(teams))
            }
            
        except Exception as e:
            logger.error(f"Error getting league statistics for {league_name}: {str(e)}")
            return None
