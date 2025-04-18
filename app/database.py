import pandas as pd
import psycopg2

from configs import load_config

conn = None


def get_connection():
    global conn
    config_db = load_config()["db"]
    if conn is None:
        conn = psycopg2.connect(
            dbname=config_db["dbname"],
            user=config_db["user"],
            password=config_db["password"],
            host=config_db["host"],
        )
        print(f'Connected to {config_db["dbname"]}')
    return conn


def get_documents_by_ids(ids) -> pd.DataFrame:
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM ds_qa WHERE id = ANY(%s)", (ids,))
        results = cursor.fetchall()
    return pd.DataFrame(
        results,
        columns=["id", "section", "subsection", "question", "answer", "answer_hash"],
    )
