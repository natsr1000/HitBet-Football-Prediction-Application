"""
Database-based repository for league data management
"""

import logging
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from models.database_models import League, Team, Match
from database_config import db_session  # Use the shared context manager here

logger = logging.getLogger(__name__)

class DatabaseLeagueRepository:
    """Database-based repository for league data management"""

    def save_league_data(self, league_name: str, match_data: pd.DataFrame) -> bool:
        """Save league match data to database"""
        with db_session() as session:
            try:
                league = session.query(League).filter_by(name=league_name).first()
                if not league:
                    country = league_name.split('-')[0] if '-' in league_name else "Unknown"
                    league = League(name=league_name, country=country)
                    session.add(league)
                    session.flush()

                teams_dict = {}
                matches_added = 0

                for _, row in match_data.iterrows():
                    try:
                        home_team_name = str(row['HomeTeam'])
                        if home_team_name not in teams_dict:
                            home_team = session.query(Team).filter_by(
                                name=home_team_name, league_name=league_name
                            ).first()
                            if not home_team:
                                home_team = Team(name=home_team_name, league_name=league_name)
                                session.add(home_team)
                                session.flush()
                            teams_dict[home_team_name] = home_team.id

                        away_team_name = str(row['AwayTeam'])
                        if away_team_name not in teams_dict:
                            away_team = session.query(Team).filter_by(
                                name=away_team_name, league_name=league_name
                            ).first()
                            if not away_team:
                                away_team = Team(name=away_team_name, league_name=league_name)
                                session.add(away_team)
                                session.flush()
                            teams_dict[away_team_name] = away_team.id

                        match_date = (
                            pd.to_datetime(row['Date']).to_pydatetime()
                            if 'Date' in row and pd.notna(row['Date'])
                            else datetime.now()
                        )

                        existing_match = session.query(Match).filter_by(
                            league_id=league.id,
                            home_team_id=teams_dict[home_team_name],
                            away_team_id=teams_dict[away_team_name],
                            date=match_date
                        ).first()
                        if existing_match:
                            continue

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

                logger.info(f"Saved {matches_added} matches for league {league_name}")
                return True

            except Exception as e:
                logger.error(f"Error saving league data for {league_name}: {str(e)}")
                return False

    def load_league_data(self, league_name: str) -> Optional[pd.DataFrame]:
        """Load league data from database as DataFrame"""
        with db_session() as session:
            try:
                league = session.query(League).filter_by(name=league_name).first()
                if not league:
                    logger.error(f"League {league_name} not found in database")
                    return None

                matches = session.query(Match).filter_by(league_id=league.id).all()
                if not matches:
                    logger.warning(f"No matches found for league {league_name}")
                    return None

                data = []
                for match in matches:
                    home_team = session.query(Team).filter_by(id=match.home_team_id).first()
                    away_team = session.query(Team).filter_by(id=match.away_team_id).first()
                    if not home_team or not away_team:
                        continue

                    row = {
                        'Date': match.date.strftime('%Y-%m-%d') if match.date else '',
                        'HomeTeam': home_team.name,
                        'AwayTeam': away_team.name,
                        'FTHG': match.home_goals,
                        'FTAG': match.away_goals,
                        'FTR': match.result,
                        'Season': match.season,
                        'HTHG': match.home_goals_ht,
                        'HTAG': match.away_goals_ht,
                        'HTR': match.result_ht
                    }

                    data.append(row)

                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} matches for league {league_name}")
                return df

            except Exception as e:
                logger.error(f"Error loading league data for {league_name}: {str(e)}")
                return None

    def get_saved_leagues(self) -> List[str]:
        """Get list of saved league names"""
        with db_session() as session:
            try:
                leagues = session.query(League.name).all()
                return [league[0] for league in leagues]
            except Exception as e:
                logger.error(f"Error getting saved leagues: {str(e)}")
                return []

    def league_exists(self, league_name: str) -> bool:
        """Check if league exists in database"""
        with db_session() as session:
            try:
                return session.query(League).filter_by(name=league_name).first() is not None
            except Exception as e:
                logger.error(f"Error checking league existence: {str(e)}")
                return False

    def delete_league(self, league_name: str) -> bool:
        """Delete league and all associated data"""
        with db_session() as session:
            try:
                league = session.query(League).filter_by(name=league_name).first()
                if not league:
                    return False
                session.delete(league)
                logger.info(f"Deleted league {league_name}")
                return True
            except Exception as e:
                logger.error(f"Error deleting league {league_name}: {str(e)}")
                return False

    def get_league_stats(self, league_name: str) -> Dict:
        """Get statistics for a specific league"""
        with db_session() as session:
            try:
                league = session.query(League).filter_by(name=league_name).first()
                if not league:
                    return {}

                match_count = session.query(Match).filter_by(league_id=league.id).count()
                team_count = session.query(Team).filter_by(league_name=league_name).count()

                first_match = session.query(Match).filter_by(league_id=league.id).order_by(Match.date.asc()).first()
                last_match = session.query(Match).filter_by(league_id=league.id).order_by(Match.date.desc()).first()

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
