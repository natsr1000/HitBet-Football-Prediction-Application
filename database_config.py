"""
Database configuration and initialization for ProphitBet application
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from models.database_models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and session lifecycle"""
    
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=StaticPool,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close database session"""
        try:
            session.close()
        except Exception as e:
            logger.error(f"Error closing database session: {str(e)}")
    
    def drop_all_tables(self):
        """Drop all tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Error dropping tables: {str(e)}")
            raise
    
    def reset_database(self):
        """Reset database by dropping and recreating all tables"""
        try:
            self.drop_all_tables()
            self.init_database()
            logger.info("Database reset completed")
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            raise

# Global database manager instance
db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def get_db_session() -> Session:
    """Get database session (convenience function)"""
    return get_db_manager().get_session()

def close_db_session(session: Session):
    """Close database session (convenience function)"""
    get_db_manager().close_session(session)