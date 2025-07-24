# models.py and league.py
from database_config import db
from datetime import datetime

class League(db.Model):
    __tablename__ = "leagues"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    country = db.Column(db.String(100), nullable=False)
    data = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<League id={self.id} name={self.name} country={self.country}>"


class Team(db.Model):
    __tablename__ = "team"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    league_name = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<Team id={self.id} name={self.name}>"


class Match(db.Model):
    __tablename__ = "match"

    id = db.Column(db.Integer, primary_key=True)
    league_id = db.Column(db.Integer, db.ForeignKey("leagues.id"), nullable=False)
    season = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, nullable=False)

    home_team_id = db.Column(db.Integer, db.ForeignKey("team.id"), nullable=False)
    away_team_id = db.Column(db.Integer, db.ForeignKey("team.id"), nullable=False)
    home_goals = db.Column(db.Integer, nullable=False)
    away_goals = db.Column(db.Integer, nullable=False)
    result = db.Column(db.String(10), nullable=False)

    home_goals_ht = db.Column(db.Integer, nullable=True)
    away_goals_ht = db.Column(db.Integer, nullable=True)
    result_ht = db.Column(db.String(10), nullable=True)

    def __repr__(self):
        return f"<Match {self.date.date()} {self.home_team_id} vs {self.away_team_id}>"


class MLModel(db.Model):
    __tablename__ = 'trained_model'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    league_id = db.Column(db.Integer, db.ForeignKey("leagues.id"), nullable=False)

    accuracy = db.Column(db.Float, nullable=True)
    training_samples = db.Column(db.Integer, nullable=True)
    test_samples = db.Column(db.Integer, nullable=True)

    features = db.Column(db.JSON, nullable=True)
    class_distribution = db.Column(db.JSON, nullable=True)
    test_report = db.Column(db.JSON, nullable=True)

    model_data = db.Column(db.Text, nullable=False)
    scaler_data = db.Column(db.Text, nullable=True)

    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<MLModel name={self.name} type={self.model_type} active={self.is_active}>"
