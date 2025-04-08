import hashlib
from pathlib import Path

import pandas as pd
from psycopg2.extras import execute_values
import numpy as np

from app import get_connection, get_faiss_index
from configs import load_config
from scripts.doc_parser import parse_documents
from scripts.embedder import get_embeddings
from scripts.utils import hash_text, insert_records, update_record
import tqdm


def sync_db():
    conn = get_connection()
    config = load_config()
    doc_files = [Path(config['doc_parsing']['input_dir']) / doc for doc in config['doc_parsing']['files']]
    faiss_index = get_faiss_index()
    recs_updated = 0
    recs_added = 0
    with conn.cursor() as cursor:
        for doc_path in doc_files:
            records = parse_documents(doc_path)
            for i, rec in tqdm.tqdm(records.iterrows()):
                new_hash = hash_text(rec['text'])
                cursor.execute("SELECT id, hash_answer FROM ds_qa WHERE section = %s "
                               "AND subsection = %s "
                               "AND question = %s", (rec['section'], rec['subsection'], rec['question']))
                row = cursor.fetchone()
                if row:
                    doc_id, existing_hash = row
                    if new_hash != existing_hash:
                        # Обновляем текст и хэш
                        embedding = get_embeddings([rec["text"]])
                        update_record(cursor, rec, doc_id)
                        recs_updated += 1
                        faiss_index.remove_ids(np.array([doc_id]))
                        faiss_index.add_with_ids(embedding, [doc_id])
                else:
                    # Добавляем новую запись
                    embedding = get_embeddings(rec["text"])
                    doc_ids = insert_records(cursor, pd.DataFrame([rec]))
                    recs_added += 1
                    faiss_index.add_with_ids(embedding, doc_ids)
    print(f'''DB synced
{recs_updated} records updated
{recs_added} records added
''')
