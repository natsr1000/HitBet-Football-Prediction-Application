"""
Database configuration and initialization for ProphitBet application using Flask-SQLAlchemy
"""

import logging
from flask_sqlalchemy import SQLAlchemy
from contextlib import contextmanager

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global SQLAlchemy instance (initialized in app.py with app context)
db = SQLAlchemy()

@contextmanager
def db_session():
    """
    Context manager for clean and safe DB session handling.
    
    Example:
        with db_session() as session:
            session.add(obj)
    """
    session = db.session
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Session rolled back due to error: {str(e)}")
        raise
    finally:
        session.close()
