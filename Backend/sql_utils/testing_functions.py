from Backend.sql_utils.sql_orm import get_user_sql_as_df, get_iris_sql_as_df, \
    upload_multiple_rows_user_event_counts_sql_as_df
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv
import os
from Backend.sql_utils.database import Base
from Backend.sql_utils import models

load_dotenv(find_dotenv())
engine = create_engine(
    os.getenv("event_count_event_type_database_connect")
)

Base.metadata.create_all(engine)

df = get_user_sql_as_df()
iris_df = get_iris_sql_as_df()

import seaborn as sns

sns_df = sns.load_dataset("iris")

[models.Iris(**x) for x in sns_df.to_dict(orient="records")]

upload_multiple_rows_user_event_counts_sql_as_df(new_users_list=[models.Iris(**x) for x in sns_df.to_dict(orient="records")])