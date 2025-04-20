from pathlib import Path

from app.database import (
    get_connection,
    get_documents_by_ids,
    insert_records,
    is_db_empty,
)
from app.embedder import get_embeddings
from app.faiss_index import add_with_ids, create_faiss_index, save_index, search_index
from configs import load_config
from scripts.doc_parser import parse_documents


def load_records_to_db_and_faiss() -> bool:
    conn = get_connection()
    config = load_config()
    recs_added = 0
    with conn:
        if not is_db_empty(conn):  # data is already loaded in db
            return True

        create_faiss_index()
        doc_files = [
            Path(config["doc_parsing"]["input_dir"]) / doc
            for doc in config["doc_parsing"]["files"]
        ]

        for doc_path in doc_files:
            records = parse_documents(doc_path)
            print("Getting Embeddings")
            embeddings = get_embeddings(records["text"].tolist())
            print("Adding data to DB")
            doc_ids = insert_records(conn, records)
            add_with_ids(embeddings, doc_ids)
            recs_added += len(records)
        save_index()

    print(f"{recs_added} records added")
    return False


def db_search(query, k=5):
    query_embedding = get_embeddings([query])
    D, I = search_index(query_embedding, k)
    results = get_documents_by_ids(I)
    return results
