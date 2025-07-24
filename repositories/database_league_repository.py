"""
Database-based repository for league data management
"""

import logging
import pandas as pd
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime

from models.database_models import League, Team, Match
from database_config import get_db_session, close_db_session

logger = logging.getLogger(__name__)

class DatabaseLeagueRepository:
    """Database-based repository for league data management"""
    
    def save_league_data(self, league_name: str, match_data: pd.DataFrame) -> bool:
        """Save league match data to database"""
        session = get_db_session()
        try:
            # Create or get league
            league = session.query(League).filter(League.name == league_name).first()
            if not league:
                # Extract country from league name
                country = league_name.split('-')[0] if '-' in league_name else "Unknown"
                league = League(name=league_name, country=country)
                session.add(league)
                session.flush()  # Get the league ID
            
            # Track teams for this league
            teams_dict = {}
            
            # Process each match
            matches_added = 0
            for _, row in match_data.iterrows():
                try:
                    # Get or create home team
                    home_team_name = str(row['HomeTeam'])
                    if home_team_name not in teams_dict:
                        home_team = session.query(Team).filter(
                            Team.name == home_team_name,
                            Team.league_name == league_name
                        ).first()
                        if not home_team:
                            home_team = Team(name=home_team_name, league_name=league_name)
                            session.add(home_team)
                            session.flush()
                        teams_dict[home_team_name] = home_team.id
                    
                    # Get or create away team
                    away_team_name = str(row['AwayTeam'])
                    if away_team_name not in teams_dict:
                        away_team = session.query(Team).filter(
                            Team.name == away_team_name,
                            Team.league_name == league_name
                        ).first()
                        if not away_team:
                            away_team = Team(name=away_team_name, league_name=league_name)
                            session.add(away_team)
                            session.flush()
                        teams_dict[away_team_name] = away_team.id
                    
                    # Parse match date
                    if 'Date' in row and pd.notna(row['Date']):
                        try:
                            match_date = pd.to_datetime(row['Date']).to_pydatetime()
                        except:
                            match_date = datetime.now()
                    else:
                        match_date = datetime.now()
                    
                    # Check if match already exists
                    existing_match = session.query(Match).filter(
                        Match.league_id == league.id,
                        Match.home_team_id == teams_dict[home_team_name],
                        Match.away_team_id == teams_dict[away_team_name],
                        Match.date == match_date
                    ).first()
                    
                    if existing_match:
                        continue  # Skip duplicate matches
                    
                    # Create match record
                    match = Match(
                        league_id=league.id,
                        date=match_date,
                        season=int(row.get('Season', 2024)),
                        home_team_id=teams_dict[home_team_name],
                        away_team_id=teams_dict[away_team_name],
                        home_goals=int(row['FTHG']),
                        away_goals=int(row['FTAG']),
                        result=str(row['FTR']),
                        home_goals_ht=int(row.get('HTHG', 0)) if pd.notna(row.get('HTHG')) else None,
                        away_goals_ht=int(row.get('HTAG', 0)) if pd.notna(row.get('HTAG')) else None,
                        result_ht=str(row.get('HTR', '')) if pd.notna(row.get('HTR')) else None
                    )
                    
                    session.add(match)
                    matches_added += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing match row: {str(e)}")
                    continue
            
            # Commit all changes
            session.commit()
            logger.info(f"Saved {matches_added} matches for league {league_name}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving league data for {league_name}: {str(e)}")
            return False
        finally:
            close_db_session(session)
    
    def load_league_data(self, league_name: str) -> Optional[pd.DataFrame]:
        """Load league data from database as DataFrame"""
        session = get_db_session()
        try:
            # Get league
            league = session.query(League).filter(League.name == league_name).first()
            if not league:
                logger.error(f"League {league_name} not found in database")
                return None
            
            # Query matches with team information
            matches = session.query(Match).filter(Match.league_id == league.id).all()
            
            if not matches:
                logger.warning(f"No matches found for league {league_name}")
                return None
            
            # Convert to DataFrame format
            data = []
            for match in matches:
                # Get team names
                home_team = session.query(Team).filter(Team.id == match.home_team_id).first()
                away_team = session.query(Team).filter(Team.id == match.away_team_id).first()
                
                if not home_team or not away_team:
                    continue
                
                row = {
                    'Date': match.date.strftime('%Y-%m-%d') if match.date else '',
                    'HomeTeam': home_team.name,
                    'AwayTeam': away_team.name,
                    'FTHG': match.home_goals,
                    'FTAG': match.away_goals,
                    'FTR': match.result,
                    'Season': match.season
                }
                
                # Add half-time data if available
                if match.home_goals_ht is not None:
                    row['HTHG'] = match.home_goals_ht
                if match.away_goals_ht is not None:
                    row['HTAG'] = match.away_goals_ht
                if match.result_ht:
                    row['HTR'] = match.result_ht
                
                data.append(row)
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} matches for league {league_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading league data for {league_name}: {str(e)}")
            return None
        finally:
            close_db_session(session)
    
    def get_saved_leagues(self) -> List[str]:
        """Get list of saved league names"""
        session = get_db_session()
        try:
            leagues = session.query(League.name).all()
            return [league[0] for league in leagues]
        except Exception as e:
            logger.error(f"Error getting saved leagues: {str(e)}")
            return []
        finally:
            close_db_session(session)
    
    def league_exists(self, league_name: str) -> bool:
        """Check if league exists in database"""
        session = get_db_session()
        try:
            league = session.query(League).filter(League.name == league_name).first()
            return league is not None
        except Exception as e:
            logger.error(f"Error checking league existence: {str(e)}")
            return False
        finally:
            close_db_session(session)
    
    def delete_league(self, league_name: str) -> bool:
        """Delete league and all associated data"""
        session = get_db_session()
        try:
            league = session.query(League).filter(League.name == league_name).first()
            if not league:
                return False
            
            # Delete league (cascade will handle matches and models)
            session.delete(league)
            session.commit()
            
            logger.info(f"Deleted league {league_name}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting league {league_name}: {str(e)}")
            return False
        finally:
            close_db_session(session)
    
    def get_league_stats(self, league_name: str) -> Dict:
        """Get statistics for a specific league"""
        session = get_db_session()
        try:
            league = session.query(League).filter(League.name == league_name).first()
            if not league:
                return {}
            
            # Count matches
            match_count = session.query(Match).filter(Match.league_id == league.id).count()
            
            # Count teams
            team_count = session.query(Team).filter(Team.league_name == league_name).count()
            
            # Get date range
            first_match = session.query(Match).filter(Match.league_id == league.id).order_by(Match.date.asc()).first()
            last_match = session.query(Match).filter(Match.league_id == league.id).order_by(Match.date.desc()).first()
            
            return {
                'name': league_name,
                'country': league.country,
                'matches': match_count,
                'teams': team_count,
                'first_match': first_match.date.strftime('%Y-%m-%d') if first_match else None,
                'last_match': last_match.date.strftime('%Y-%m-%d') if last_match else None,
                'created_at': league.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error getting league stats for {league_name}: {str(e)}")
            return {}
        finally:
            close_db_session(session)