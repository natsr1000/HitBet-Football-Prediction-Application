"""
Database configuration and initialization for ProphitBet application using Flask-SQLAlchemy
"""

import logging
from flask_sqlalchemy import SQLAlchemy
from contextlib import contextmanager

# Logger setup
logger = logging.getLogger(__name__)

# Create a global SQLAlchemy instance
db = SQLAlchemy()

@contextmanager
def db_session():
    """
    Context manager for clean and safe DB session handling.
    Automatically commits or rolls back as needed.
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
