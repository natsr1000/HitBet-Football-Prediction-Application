"""
Database configuration and initialization for ProphitBet application
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from sqlalchemy.pool import StaticPool

# Logging setup
logger = logging.getLogger(__name__)

# Base for all ORM models
Base = declarative_base()

class DatabaseManager:
    """Manages database connections and session lifecycle"""

    def __init__(self):
        self.database_url = os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")

        # Create engine
        self.engine = create_engine(
            self.database_url,
            poolclass=StaticPool,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False  # Set to True for SQL debugging
        )

        # Session factory (scoped for thread safety in Flask)
        self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))

    def init_database(self):
        """Create all tables based on ORM models"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise

    def drop_all_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Error dropping database tables: {str(e)}")
            raise

    def reset_database(self):
        """Drop and recreate all tables"""
        self.drop_all_tables()
        self.init_database()

    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()

    def close_session(self, session):
        """Close database session"""
        try:
            session.close()
        except Exception as e:
            logger.error(f"Error closing session: {str(e)}")


# Global instance (lazy initialized)
_db_manager = None

def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def get_db_session():
    """Convenience function to get a new DB session"""
    return get_db_manager().get_session()

def close_db_session(session):
    """Convenience function to close a DB session"""
    get_db_manager().close_session(session)
