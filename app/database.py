import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from app.logger import setup_logger
from configs import load_config

conn = None
logger = setup_logger(__name__)


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
        logger.info(f'Connected to {config_db["dbname"]}')
    return conn


def get_documents_by_ids(ids) -> pd.DataFrame:
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM ds_qa WHERE id = ANY(%s)", (ids,))
            results = cursor.fetchall()
    except psycopg2.Error as e:
        logger.error("Error while reading db: %s", e)
        raise
    return pd.DataFrame(
        results,
        columns=["id", "section", "subsection", "question", "answer", "answer_hash"],
    )


def is_db_empty(connection) -> bool:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM ds_qa LIMIT 1")
            result = cursor.fetchall()
    except psycopg2.Error as e:
        logger.error("Error while checking db: %s", e)
        raise

    return not bool(result)


def insert_records(connection, records, batch=20):
    records_to_insert = [
        (
            rec["section"],
            rec["subsection"],
            rec["question"],
            rec["answer"],
            rec["hash_answer"],
        )
        for _, rec in records.iterrows()
    ]
    doc_ids = []

    try:
        with connection.cursor() as cursor:
            for i in range(0, len(records), batch):
                execute_values(
                    cursor,
                    f"INSERT INTO ds_qa (section, subsection, question, answer, hash_answer) "
                    "VALUES %s RETURNING id",
                    records_to_insert[i : i + batch],
                )
                doc_ids.extend([row[0] for row in cursor.fetchall()])
            logger.info("Added %d records to ds_qa.", len(records))
    except psycopg2.Error as e:
        logger.error("Error while inserting records: %s", e)
        raise

    return doc_ids


def delete_records(connection, ids):
    try:
        with connection.cursor() as cursor:
            cursor.executemany(f"DELETE FROM ds_qa WHERE id = %s", [(i,) for i in ids])
            logger.info("Deleted %d records from ds_qa.", len(ids))
    except psycopg2.Error as e:
        logger.error("Error while deleting records: %s", e)
        raise


def update_records(connection, records, batch=20):
    records_to_update = [
        (
            rec["section"],
            rec["subsection"],
            rec["question"],
            rec["answer"],
            rec["hash_answer"],
            rec["id"],
        )
        for _, rec in records.iterrows()
    ]
    query = """
        UPDATE ds_qa
        SET section = %s,
            subsection = %s,
            question = %s,
            answer = %s,
            hash_answer = %s
        WHERE id = %s
    """
    try:
        with connection.cursor() as cursor:
            for i in range(0, len(records_to_update), batch):
                cursor.executemany(query, records_to_update[i : i + batch])
            logger.info("Updated %d records in ds_qa.", len(records))
    except psycopg2.Error as e:
        logger.error("Error while updating records: %s", e)
        raise


def get_all_questions(connection, section):
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, section, subsection, question, answer, hash_answer FROM ds_qa WHERE section = %s",
                (section,),
            )
            results = cursor.fetchall()
    except psycopg2.Error as e:
        logger.error("Error while reading db: %s", e)
        raise
    return pd.DataFrame(
        results,
        columns=["id", "section", "subsection", "question", "answer", "hash_answer"],
    )


def get_all_sections():
    try:
        connection = get_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT section FROM ds_qa")
            results = cursor.fetchall()
    except psycopg2.Error as e:
        logger.error("Error while reading db: %s", e)
        return []
    return [i[0] for i in results]


def get_random_questions(section: str, n: int = 3):
    query = f"""SELECT section, subsection, question, answer 
                FROM ds_qa {f'WHERE section = \'{section}\'' if section else ''}
                ORDER BY RANDOM()
                LIMIT {n}
                """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
    except psycopg2.Error as e:
        logger.error("Error while reading db: %s", e)
        raise
    return pd.DataFrame(
        results, columns=["section", "subsection", "question", "answer"]
    )
