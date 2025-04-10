from pathlib import Path

from app.database import get_connection, get_documents_by_ids
from app.embedder import get_embeddings
from app.faiss_index import search_index, add_with_ids
from configs import load_config
from scripts.doc_parser import parse_documents
from scripts.utils import insert_records


def load_records_to_db_and_faiss():
    conn = get_connection()
    config = load_config()
    doc_files = [Path(config['doc_parsing']['input_dir']) / doc for doc in config['doc_parsing']['files']]

    with conn.cursor() as cursor:
        for doc_path in doc_files:
            records = parse_documents(doc_path)
            print('Getting Embeddings')
            embeddings = get_embeddings(records["text"].tolist())
            print('Adding data to DB')
            doc_ids = insert_records(cursor, records)
            add_with_ids(embeddings, doc_ids)
    conn.commit()


def db_search(query, k=5):
    query_embedding = get_embeddings([query])
    D, I = search_index(query_embedding, k)
    results = get_documents_by_ids(I)
    return results
