"""
Utilities for team name mapping and fuzzy matching
"""

import json
import os
import logging
from typing import Optional, List, Dict
import pandas as pd
# from fuzzywuzzy import fuzz, process  # Temporarily disabled

from config import TEAM_MAPPINGS_FILE

logger = logging.getLogger(__name__)

class TeamMapper:
    """Handles team name mapping and fuzzy matching"""
    
    def __init__(self):
        self.mappings = self._load_mappings()
    
    def _load_mappings(self) -> Dict[str, str]:
        """Load team name mappings from file"""
        try:
            if os.path.exists(TEAM_MAPPINGS_FILE):
                with open(TEAM_MAPPINGS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading team mappings: {str(e)}")
        
        return {}
    
    def _save_mappings(self):
        """Save team name mappings to file"""
        try:
            with open(TEAM_MAPPINGS_FILE, 'w') as f:
                json.dump(self.mappings, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving team mappings: {str(e)}")
    
    def map_team_name(self, team_name: str, available_teams: List[str], 
                      threshold: int = 80) -> Optional[str]:
        """
        Map a team name to the closest match in available teams
        
        Args:
            team_name: The team name to map
            available_teams: List of available team names
            threshold: Minimum similarity score (0-100)
        
        Returns:
            Mapped team name or None if no good match found
        """
        # Check if exact match exists
        if team_name in available_teams:
            return team_name
        
        # Check cached mappings
        if team_name in self.mappings:
            mapped_name = self.mappings[team_name]
            if mapped_name in available_teams:
                return mapped_name
        
        # Fuzzy matching
        best_match, score = process.extractOne(
            team_name, 
            available_teams, 
            scorer=fuzz.ratio
        )
        
        if score >= threshold:
            # Cache the mapping
            self.mappings[team_name] = best_match
            self._save_mappings()
            logger.info(f"Mapped '{team_name}' to '{best_match}' (score: {score})")
            return best_match
        
        # Try partial matching for better results
        best_partial, partial_score = process.extractOne(
            team_name,
            available_teams,
            scorer=fuzz.partial_ratio
        )
        
        if partial_score >= threshold:
            self.mappings[team_name] = best_partial
            self._save_mappings()
            logger.info(f"Mapped '{team_name}' to '{best_partial}' (partial score: {partial_score})")
            return best_partial
        
        logger.warning(f"No good match found for '{team_name}' (best: '{best_match}', score: {score})")
        return None
    
    def get_similar_teams(self, team_name: str, available_teams: List[str], 
                         limit: int = 5) -> List[Dict[str, float]]:
        """Get list of similar team names with scores"""
        try:
            matches = process.extract(
                team_name,
                available_teams,
                scorer=fuzz.ratio,
                limit=limit
            )
            
            return [{'name': match[0], 'score': match[1]} for match in matches]
            
        except Exception as e:
            logger.error(f"Error getting similar teams for '{team_name}': {str(e)}")
            return []

# Global team mapper instance
_team_mapper = TeamMapper()

def map_team_names(team_name: str, league_data: pd.DataFrame, 
                   threshold: int = 80) -> Optional[str]:
    """
    Convenience function to map team names using league data
    
    Args:
        team_name: Team name to map
        league_data: DataFrame containing league data
        threshold: Minimum similarity score
    
    Returns:
        Mapped team name or None
    """
    try:
        # Get all unique team names from the league data
        home_teams = set(league_data['HomeTeam'].unique())
        away_teams = set(league_data['AwayTeam'].unique())
        available_teams = sorted(list(home_teams | away_teams))
        
        return _team_mapper.map_team_name(team_name, available_teams, threshold)
        
    except Exception as e:
        logger.error(f"Error mapping team name '{team_name}': {str(e)}")
        return None

def get_team_suggestions(team_name: str, league_data: pd.DataFrame, 
                        limit: int = 5) -> List[Dict[str, float]]:
    """
    Get team name suggestions for user input
    
    Args:
        team_name: Partial or incorrect team name
        league_data: DataFrame containing league data
        limit: Maximum number of suggestions
    
    Returns:
        List of suggested team names with similarity scores
    """
    try:
        # Get all unique team names from the league data
        home_teams = set(league_data['HomeTeam'].unique())
        away_teams = set(league_data['AwayTeam'].unique())
        available_teams = sorted(list(home_teams | away_teams))
        
        return _team_mapper.get_similar_teams(team_name, available_teams, limit)
        
    except Exception as e:
        logger.error(f"Error getting team suggestions for '{team_name}': {str(e)}")
        return []

def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name for better matching
    
    Args:
        team_name: Raw team name
    
    Returns:
        Normalized team name
    """
    try:
        # Basic normalization
        normalized = team_name.strip()
        
        # Remove common suffixes/prefixes
        suffixes_to_remove = [' FC', ' F.C.', ' United', ' City', ' Town', ' CF', ' AFC', ' LFC']
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
                break
        
        # Handle common abbreviations
        abbreviations = {
            'Man Utd': 'Manchester United',
            'Man City': 'Manchester City',
            'Spurs': 'Tottenham',
            'Arsenal': 'Arsenal',
            'Chelsea': 'Chelsea',
            'Liverpool': 'Liverpool'
        }
        
        if normalized in abbreviations:
            normalized = abbreviations[normalized]
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error normalizing team name '{team_name}': {str(e)}")
        return team_name