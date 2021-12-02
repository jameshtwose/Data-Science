from Backend.sql_utils import models
from Backend.sql_utils.database import session_scope, Base, engine
import pandas as pd


# Base.metadata.create_all(engine)


def get_user_sql_as_df():
    with session_scope() as session:
        new_query = session.query(models.User)
        return pd.read_sql(new_query.statement, new_query.session.bind)


def get_iris_sql_as_df():
    with session_scope() as session:
        new_query = session.query(models.Iris)
        return pd.read_sql(new_query.statement, new_query.session.bind)


# def get_user_event_counts_sql_as_df():
#     with session_scope() as session:
#         new_query = session.query(models.UserEventCounts)
#         return pd.read_sql(new_query.statement, new_query.session.bind)


def upload_row_user_event_counts_sql_as_df(new_user):
    with session_scope() as session:
        session.add(new_user)
        session.commit()


def upload_multiple_rows_user_event_counts_sql_as_df(new_users_list):
    with session_scope() as session:
        session.add_all(new_users_list)
        session.commit()

