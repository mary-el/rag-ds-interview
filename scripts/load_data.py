from pathlib import Path

from app.database import get_connection, get_documents_by_ids
from app.embedder import get_embeddings
from app.faiss_index import search_index, add_with_ids, save_index, create_faiss_index
from configs import load_config
from scripts.doc_parser import parse_documents
from app.utils import insert_records, is_db_empty


def load_records_to_db_and_faiss():
    conn = get_connection()
    config = load_config()

    with conn.cursor() as cursor:
        if not is_db_empty(cursor):   # data is already loaded in db
            return

        create_faiss_index()
        doc_files = [Path(config['doc_parsing']['input_dir']) / doc for doc in config['doc_parsing']['files']]

        for doc_path in doc_files:
            records = parse_documents(doc_path)
            print('Getting Embeddings')
            embeddings = get_embeddings(records["text"].tolist())
            print('Adding data to DB')
            doc_ids = insert_records(cursor, records)
            add_with_ids(embeddings, doc_ids)
    conn.commit()
    save_index()


def db_search(query, k=5):
    query_embedding = get_embeddings([query])
    D, I = search_index(query_embedding, k)
    results = get_documents_by_ids(I)
    return results
