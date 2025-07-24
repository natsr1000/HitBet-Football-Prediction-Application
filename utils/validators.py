"""
Validation utilities for input data and user requests
"""

import re
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def validate_fixture_input(fixture_data: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate fixture data format
    
    Args:
        fixture_data: Dictionary containing fixture information
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        required_fields = ['home_team', 'away_team']
        
        # Check required fields
        for field in required_fields:
            if field not in fixture_data:
                return False, f"Missing required field: {field}"
            
            if not fixture_data[field] or not isinstance(fixture_data[field], str):
                return False, f"Invalid {field}: must be a non-empty string"
        
        # Validate team names
        home_team = fixture_data['home_team'].strip()
        away_team = fixture_data['away_team'].strip()
        
        if len(home_team) < 2:
            return False, "Home team name too short"
        
        if len(away_team) < 2:
            return False, "Away team name too short"
        
        if home_team.lower() == away_team.lower():
            return False, "Home and away teams cannot be the same"
        
        # Validate date if provided
        if 'date' in fixture_data and fixture_data['date']:
            date_str = fixture_data['date']
            if not _validate_date_format(date_str):
                return False, f"Invalid date format: {date_str}"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating fixture input: {str(e)}")
        return False, f"Validation error: {str(e)}"

def validate_csv_format(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate CSV file format for predictions
    
    Args:
        df: Pandas DataFrame from CSV file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        required_columns = ['HomeTeam', 'AwayTeam']
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for empty DataFrame
        if len(df) == 0:
            return False, "CSV file is empty"
        
        # Validate each row
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Check for missing values
            if pd.isna(home_team) or pd.isna(away_team):
                return False, f"Missing team names in row {idx + 1}"
            
            # Convert to string and validate
            home_team = str(home_team).strip()
            away_team = str(away_team).strip()
            
            if len(home_team) < 2:
                return False, f"Invalid home team name in row {idx + 1}: '{home_team}'"
            
            if len(away_team) < 2:
                return False, f"Invalid away team name in row {idx + 1}: '{away_team}'"
            
            if home_team.lower() == away_team.lower():
                return False, f"Home and away teams are the same in row {idx + 1}"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating CSV format: {str(e)}")
        return False, f"CSV validation error: {str(e)}"

def validate_model_name(model_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate model name format
    
    Args:
        model_name: Name for the model
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not model_name or not isinstance(model_name, str):
            return False, "Model name must be a non-empty string"
        
        model_name = model_name.strip()
        
        if len(model_name) < 3:
            return False, "Model name must be at least 3 characters long"
        
        if len(model_name) > 50:
            return False, "Model name must be less than 50 characters"
        
        # Check for valid characters (alphanumeric, spaces, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', model_name):
            return False, "Model name can only contain letters, numbers, spaces, hyphens, and underscores"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating model name: {str(e)}")
        return False, f"Model name validation error: {str(e)}"

def validate_league_name(league_name: str, available_leagues: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate league name against available leagues
    
    Args:
        league_name: Name of the league
        available_leagues: List of available league names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not league_name or not isinstance(league_name, str):
            return False, "League name must be a non-empty string"
        
        league_name = league_name.strip()
        
        if league_name not in available_leagues:
            return False, f"League '{league_name}' not found in available leagues"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating league name: {str(e)}")
        return False, f"League name validation error: {str(e)}"

def _validate_date_format(date_str: str) -> bool:
    """
    Validate date string format
    
    Args:
        date_str: Date string to validate
    
    Returns:
        True if valid date format, False otherwise
    """
    try:
        # Common date formats
        date_patterns = [
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # DD/MM/YYYY or DD-MM-YYYY
            r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',    # YYYY/MM/DD or YYYY-MM-DD
            r'^\d{1,2}\s+\w+\s+\d{4}$',          # DD Month YYYY
            r'^\w+\s+\d{1,2},?\s+\d{4}$'         # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, date_str):
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error validating date format: {str(e)}")
        return False

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    try:
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        
        # Limit length
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'unnamed_file'
        
        return sanitized
        
    except Exception as e:
        logger.error(f"Error sanitizing filename: {str(e)}")
        return 'unnamed_file'

def validate_prediction_confidence(confidence: float) -> bool:
    """
    Validate prediction confidence value
    
    Args:
        confidence: Confidence score (should be between 0 and 1)
    
    Returns:
        True if valid confidence value, False otherwise
    """
    try:
        return isinstance(confidence, (int, float)) and 0 <= confidence <= 1
    except:
        return False
