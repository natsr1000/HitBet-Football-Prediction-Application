"""
Database configuration and initialization for ProphitBet application using Flask-SQLAlchemy.
"""

import logging
from flask_sqlalchemy import SQLAlchemy
from contextlib import contextmanager

# Set up module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global SQLAlchemy instance (initialized in app.py)
db = SQLAlchemy()

@contextmanager
def db_session():
    """
    Context manager for clean and safe DB session handling.

    Example:
        with db_session() as session:
            session.add(some_model)
    """
    session = db.session
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session rolled back due to error: {e}")
        raise
    finally:
        session.close()
