"""
Database models for ProphitBet application using SQLAlchemy
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()

class League(Base):
    """Model for storing league information"""
    __tablename__ = 'leagues'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    country = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    matches = relationship("Match", back_populates="league", cascade="all, delete-orphan")
    models = relationship("MLModel", back_populates="league", cascade="all, delete-orphan")

class Team(Base):
    """Model for storing team information"""
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    league_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")

class Match(Base):
    """Model for storing match data"""
    __tablename__ = 'matches'
    
    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey('leagues.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    season = Column(Integer, nullable=False)
    
    # Team information
    home_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    away_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    
    # Match results
    home_goals = Column(Integer, nullable=False)
    away_goals = Column(Integer, nullable=False)
    result = Column(String(1), nullable=False)  # H, D, A
    
    # Half-time scores
    home_goals_ht = Column(Integer)
    away_goals_ht = Column(Integer)
    result_ht = Column(String(1))  # H, D, A
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    league = relationship("League", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

class MLModel(Base):
    """Model for storing trained ML models"""
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    model_type = Column(String(50), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.id'), nullable=False)
    
    # Model metadata
    accuracy = Column(Float)
    training_samples = Column(Integer)
    test_samples = Column(Integer)
    features = Column(Integer)
    class_distribution = Column(JSON)
    test_report = Column(JSON)
    
    # Model storage
    model_data = Column(Text)  # Serialized model data
    scaler_data = Column(Text)  # Serialized scaler data
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Relationships
    league = relationship("League", back_populates="models")
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")

class Prediction(Base):
    """Model for storing prediction results"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'), nullable=False)
    
    # Match information
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    
    # Prediction results
    predicted_result = Column(String(10), nullable=False)  # Home Win, Draw, Away Win
    confidence = Column(Float, nullable=False)
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("MLModel", back_populates="predictions")

class ScrapedFixture(Base):
    """Model for storing scraped fixture data"""
    __tablename__ = 'scraped_fixtures'
    
    id = Column(Integer, primary_key=True)
    league_name = Column(String(100), nullable=False)
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    match_date = Column(DateTime, nullable=False)
    
    # Scraping metadata
    source_url = Column(String(500))
    scraped_at = Column(DateTime, default=datetime.utcnow)
    is_processed = Column(Boolean, default=False)
    
    # Match result (if available after match)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    result = Column(String(1))  # H, D, A

class SystemLog(Base):
    """Model for storing system logs and events"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    level = Column(String(10), nullable=False)  # INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    module = Column(String(50))
    function = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String(50))  # For future user tracking
    
    # Additional context data
    context_data = Column(JSON)