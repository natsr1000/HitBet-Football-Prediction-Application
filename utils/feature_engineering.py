"""
Feature engineering utilities for football prediction
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

def compute_team_statistics(data: pd.DataFrame, team_name: str, 
                          num_recent_matches: int = 10) -> Optional[np.ndarray]:
    """
    Compute comprehensive team statistics from historical data
    
    Returns 16 features:
    - Wins, Draws, Losses (home and away)
    - Goals For, Goals Against (home and away)
    - Win/Draw/Loss percentages
    - Average goals per game
    - Form (recent performance)
    """
    try:
        if len(data) == 0:
            return None
        
        # Filter matches for the team
        home_matches = data[data['HomeTeam'] == team_name].copy()
        away_matches = data[data['AwayTeam'] == team_name].copy()
        
        # Get recent matches
        recent_home = home_matches.tail(num_recent_matches // 2)
        recent_away = away_matches.tail(num_recent_matches // 2)
        
        # Home statistics
        home_wins = (recent_home['FTR'] == 'H').sum()
        home_draws = (recent_home['FTR'] == 'D').sum()
        home_losses = (recent_home['FTR'] == 'A').sum()
        home_goals_for = recent_home['FTHG'].sum()
        home_goals_against = recent_home['FTAG'].sum()
        
        # Away statistics
        away_wins = (recent_away['FTR'] == 'A').sum()
        away_draws = (recent_away['FTR'] == 'D').sum()
        away_losses = (recent_away['FTR'] == 'H').sum()
        away_goals_for = recent_away['FTAG'].sum()
        away_goals_against = recent_away['FTHG'].sum()
        
        # Overall statistics
        total_matches = len(recent_home) + len(recent_away)
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_losses = home_losses + away_losses
        total_goals_for = home_goals_for + away_goals_for
        total_goals_against = home_goals_against + away_goals_against
        
        if total_matches == 0:
            return np.zeros(16)
        
        # Calculate percentages and averages
        win_percentage = total_wins / total_matches
        draw_percentage = total_draws / total_matches
        loss_percentage = total_losses / total_matches
        avg_goals_for = total_goals_for / total_matches
        avg_goals_against = total_goals_against / total_matches
        
        # Form (points from recent matches: 3 for win, 1 for draw, 0 for loss)
        recent_points = (total_wins * 3) + (total_draws * 1)
        form = recent_points / (total_matches * 3) if total_matches > 0 else 0
        
        # Compile features
        features = np.array([
            total_wins,           # 0: Total wins
            total_draws,          # 1: Total draws
            total_losses,         # 2: Total losses
            home_wins,            # 3: Home wins
            home_draws,           # 4: Home draws
            home_losses,          # 5: Home losses
            away_wins,            # 6: Away wins
            away_draws,           # 7: Away draws
            away_losses,          # 8: Away losses
            total_goals_for,      # 9: Total goals for
            total_goals_against,  # 10: Total goals against
            win_percentage,       # 11: Win percentage
            draw_percentage,      # 12: Draw percentage
            loss_percentage,      # 13: Loss percentage
            avg_goals_for,        # 14: Average goals for
            form                  # 15: Recent form
        ], dtype=np.float32)
        
        return features
        
    except Exception as e:
        logger.error(f"Error computing team statistics for {team_name}: {str(e)}")
        return None

def create_match_features(home_stats: np.ndarray, away_stats: np.ndarray) -> np.ndarray:
    """
    Create match-level features by combining home and away team statistics
    """
    try:
        # Direct concatenation of home and away features
        direct_features = np.concatenate([home_stats, away_stats])
        
        # Differential features (home - away)
        diff_features = home_stats - away_stats
        
        # Ratio features (home / away, with safety for division by zero)
        ratio_features = np.divide(home_stats, away_stats, 
                                 out=np.ones_like(home_stats), 
                                 where=away_stats!=0)
        
        # Combine all features
        all_features = np.concatenate([direct_features, diff_features, ratio_features])
        
        return all_features
        
    except Exception as e:
        logger.error(f"Error creating match features: {str(e)}")
        return np.concatenate([home_stats, away_stats])

def get_feature_names() -> List[str]:
    """Get descriptive names for all features"""
    base_features = [
        'total_wins', 'total_draws', 'total_losses',
        'home_wins', 'home_draws', 'home_losses',
        'away_wins', 'away_draws', 'away_losses',
        'total_goals_for', 'total_goals_against',
        'win_percentage', 'draw_percentage', 'loss_percentage',
        'avg_goals_for', 'form'
    ]
    
    feature_names = []
    
    # Home team features
    feature_names.extend([f'home_{feat}' for feat in base_features])
    
    # Away team features
    feature_names.extend([f'away_{feat}' for feat in base_features])
    
    # Differential features
    feature_names.extend([f'diff_{feat}' for feat in base_features])
    
    # Ratio features
    feature_names.extend([f'ratio_{feat}' for feat in base_features])
    
    return feature_names

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to handle different scales"""
    try:
        # Handle NaN and infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Simple min-max normalization for each feature
        feature_min = np.min(features, axis=0)
        feature_max = np.max(features, axis=0)
        
        # Avoid division by zero
        feature_range = feature_max - feature_min
        feature_range[feature_range == 0] = 1
        
        normalized = (features - feature_min) / feature_range
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error normalizing features: {str(e)}")
        return features
