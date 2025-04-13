import pandas as pd
import psycopg2

from configs import CONFIG

conn = None


def get_connection():
    global conn
    if conn is None:
        conn = psycopg2.connect(dbname=CONFIG['dbname'], user=CONFIG['user'], password=CONFIG['password'],
                                host=CONFIG['host'])
        print(f'Connected to {CONFIG["dbname"]}')
    return conn


def get_documents_by_ids(ids) -> pd.DataFrame:
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM ds_qa WHERE id = ANY(%s)", (ids,))
        results = cursor.fetchall()
    return pd.DataFrame(results, columns=['id', 'section', 'subsection', 'question', 'answer', 'answer_hash'])
