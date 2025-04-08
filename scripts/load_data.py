from pathlib import Path

from app import get_connection, get_faiss_index
from configs import load_config
from scripts.doc_parser import parse_documents
from scripts.embedder import get_embeddings
from scripts.utils import insert_records


def load_records_to_db_and_faiss():
    conn = get_connection()
    config = load_config()
    faiss_index = get_faiss_index()
    doc_files = [Path(config['doc_parsing']['input_dir']) / doc for doc in config['doc_parsing']['files']]

    with conn.cursor() as cursor:
        for doc_path in doc_files:
            records = parse_documents(doc_path)
            print('Getting Embeddings')
            embeddings = get_embeddings(records["text"].tolist())
            print('Adding data to DB')
            doc_ids = insert_records(cursor, records)
            faiss_index.add_with_ids(embeddings, doc_ids)
    conn.commit()
    conn.close()


def db_search(query, k=5):
    conn = get_connection()
    faiss_index = get_faiss_index()
    query_embedding = get_embeddings([query])
    D, I = faiss_index.search(query_embedding, k)
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM ds_qa WHERE id = ANY(%s)", (I[0].tolist(),))
        results = cursor.fetchall()
    return results
