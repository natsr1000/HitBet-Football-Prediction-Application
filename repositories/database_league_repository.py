"""
Database-based repository for league data management.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

from models.database_models import League, Team, Match
from database_config import db_session

logger = logging.getLogger(__name__)


class DatabaseLeagueRepository:
    """Repository for managing league data in the database."""

    def _get_or_create_league(self, session, league_name: str) -> League:
        league = session.query(League).filter_by(name=league_name).first()
        if not league:
            country = league_name.split('-')[0] if '-' in league_name else "Unknown"
            league = League(name=league_name, country=country)
            session.add(league)
            session.flush()
        return league

    def _get_or_create_team(self, session, team_name: str, league_name: str, cache: dict) -> int:
        if team_name in cache:
            return cache[team_name]

        team = session.query(Team).filter_by(name=team_name, league_name=league_name).first()
        if not team:
            team = Team(name=team_name, league_name=league_name)
            session.add(team)
            session.flush()

        cache[team_name] = team.id
        return team.id

    def save_league_data(self, league_name: str, match_data: pd.DataFrame) -> bool:
        """Save league match data to the database."""
        with db_session() as session:
            try:
                league = self._get_or_create_league(session, league_name)
                teams_cache = {}
                matches_added = 0

                for _, row in match_data.iterrows():
                    try:
                        home_team_id = self._get_or_create_team(session, str(row['HomeTeam']), league_name, teams_cache)
                        away_team_id = self._get_or_create_team(session, str(row['AwayTeam']), league_name, teams_cache)
                        match_date = pd.to_datetime(row.get('Date')).to_pydatetime() if pd.notna(row.get('Date')) else datetime.now()

                        if session.query(Match).filter_by(
                            league_id=league.id,
                            home_team_id=home_team_id,
                            away_team_id=away_team_id,
                            date=match_date
                        ).first():
                            continue

                        match = Match(
                            league_id=league.id,
                            date=match_date,
                            season=int(row.get('Season', 2024)),
                            home_team_id=home_team_id,
                            away_team_id=away_team_id,
                            home_goals=int(row.get('FTHG', 0)),
                            away_goals=int(row.get('FTAG', 0)),
                            result=str(row.get('FTR', '')),
                            home_goals_ht=int(row.get('HTHG', 0)) if pd.notna(row.get('HTHG')) else None,
                            away_goals_ht=int(row.get('HTAG', 0)) if pd.notna(row.get('HTAG')) else None,
                            result_ht=str(row.get('HTR', '')) if pd.notna(row.get('HTR')) else None
                        )

                        session.add(match)
                        matches_added += 1

                    except Exception as e:
                        logger.warning(f"Error processing match row: {e}")
                        continue

                logger.info(f"Saved {matches_added} matches for league '{league_name}'")
                return True

            except Exception as e:
                logger.error(f"Failed to save league data for '{league_name}': {e}")
                return False

    def load_league_data(self, league_name: str) -> Optional[pd.DataFrame]:
        """Load league match data as a DataFrame."""
        with db_session() as session:
            try:
                league = session.query(League).filter_by(name=league_name).first()
                if not league:
                    logger.warning(f"League '{league_name}' not found.")
                    return None

                matches = session.query(Match).filter_by(league_id=league.id).all()
                if not matches:
                    logger.info(f"No matches found for league '{league_name}'.")
                    return None

                team_map = {team.id: team.name for team in session.query(Team).filter_by(league_name=league_name).all()}

                data = [
                    {
                        'Date': match.date.strftime('%Y-%m-%d') if match.date else '',
                        'HomeTeam': team_map.get(match.home_team_id, ''),
                        'AwayTeam': team_map.get(match.away_team_id, ''),
                        'FTHG': match.home_goals,
                        'FTAG': match.away_goals,
                        'FTR': match.result,
                        'Season': match.season,
                        'HTHG': match.home_goals_ht,
                        'HTAG': match.away_goals_ht,
                        'HTR': match.result_ht
                    }
                    for match in matches
                ]

                return pd.DataFrame(data)

            except Exception as e:
                logger.error(f"Failed to load league data for '{league_name}': {e}")
                return None

    def get_saved_leagues(self) -> List[str]:
        """Return list of saved league names."""
        with db_session() as session:
            try:
                return [name for (name,) in session.query(League.name).all()]
            except Exception as e:
                logger.error(f"Failed to get saved leagues: {e}")
                return []

    def league_exists(self, league_name: str) -> bool:
        """Check if a league exists in the database."""
        with db_session() as session:
            try:
                return session.query(League.id).filter_by(name=league_name).first() is not None
            except Exception as e:
                logger.error(f"Error checking existence of league '{league_name}': {e}")
                return False

    def delete_league(self, league_name: str) -> bool:
        """Delete a league and its data from the database."""
        with db_session() as session:
            try:
                league = session.query(League).filter_by(name=league_name).first()
                if not league:
                    return False
                session.delete(league)
                logger.info(f"Deleted league '{league_name}'")
                return True
            except Exception as e:
                logger.error(f"Failed to delete league '{league_name}': {e}")
                return False

    def get_league_stats(self, league_name: str) -> Dict:
        """Return league statistics (matches, teams, dates)."""
        with db_session() as session:
            try:
                league = session.query(League).filter_by(name=league_name).first()
                if not league:
                    return {}

                match_q = session.query(Match).filter_by(league_id=league.id)
                team_q = session.query(Team).filter_by(league_name=league_name)

                first_match = match_q.order_by(Match.date.asc()).first()
                last_match = match_q.order_by(Match.date.desc()).first()

                return {
                    'name': league_name,
                    'country': league.country,
                    'matches': match_q.count(),
                    'teams': team_q.count(),
                    'first_match': first_match.date.strftime('%Y-%m-%d') if first_match else None,
                    'last_match': last_match.date.strftime('%Y-%m-%d') if last_match else None,
                    'created_at': league.created_at.strftime('%Y-%m-%d %H:%M:%S') if league.created_at else None
                }

            except Exception as e:
                logger.error(f"Failed to get stats for league '{league_name}': {e}")
                return {}
