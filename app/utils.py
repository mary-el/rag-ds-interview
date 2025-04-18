import hashlib
from enum import Enum

from psycopg2.extras import execute_values


class Mode(Enum):
    QA = "qa"
    QUIZ = "quiz"


def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_db_empty(cursor) -> bool:
    cursor.execute("SELECT id FROM ds_qa LIMIT 1")
    result = cursor.fetchall()
    return not bool(result)


def insert_records(cursor, records, batch=20):
    records_to_insert = [
        (
            rec["section"],
            rec["subsection"],
            rec["question"],
            rec["answer"],
            hash_text(rec["text"]),
        )
        for _, rec in records.iterrows()
    ]
    doc_ids = []
    for i in range(0, len(records), batch):
        execute_values(
            cursor,
            f"INSERT INTO ds_qa (section, subsection, question, answer, hash_answer) "
            "VALUES %s RETURNING id",
            records_to_insert[i : i + batch],
        )
        doc_ids.extend([row[0] for row in cursor.fetchall()])
    return doc_ids


def update_record(cursor, rec, doc_id):
    cursor.execute(
        """
        UPDATE ds_qa
        SET section = %s, subsection = %s, answer = %s, question = %s, hash_answer = %s
        WHERE id = %s
    """,
        (
            rec["section"],
            rec["subsection"],
            rec["answer"],
            rec["question"],
            hash_text((rec["text"])),
            doc_id,
        ),
    )
