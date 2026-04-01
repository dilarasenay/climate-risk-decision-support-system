import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_TABLE = os.getenv("DB_TABLE")

def get_engine():
    connection_string = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)

def load_data():
    engine = get_engine()
    query = f"SELECT * FROM {DB_TABLE}"
    return pd.read_sql(query, engine)

def insert_row(row_dict: dict):
    engine = get_engine()
    df = pd.DataFrame([row_dict])
    df.to_sql(DB_TABLE, con=engine, if_exists="append", index=False)