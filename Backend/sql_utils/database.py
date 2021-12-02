from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv, find_dotenv
import os
from contextlib import contextmanager


load_dotenv(find_dotenv())

engine = create_engine(
    os.getenv("event_count_event_type_database_connect")
)

Base = declarative_base()

Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


# def run_my_program():
#     with session_scope() as session:
#         ThingOne().go(session)
#         ThingTwo().go(session)