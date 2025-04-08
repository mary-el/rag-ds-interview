import psycopg2

from configs import DB_CONFIG

conn = None


def get_connection():
    global conn
    if conn is None:
        conn = psycopg2.connect(dbname=DB_CONFIG['dbname'], user=DB_CONFIG['user'], password=DB_CONFIG['password'],
                                host=DB_CONFIG['host'])
        print(f'Connected to {DB_CONFIG["dbname"]}')
    return conn
