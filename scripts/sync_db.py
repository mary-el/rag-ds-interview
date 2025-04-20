from pathlib import Path

import numpy as np
import pandas as pd

from app import get_connection, get_faiss_index
from app.database import (
    delete_records,
    get_all_questions,
    insert_records,
    update_records,
)
from app.embedder import get_embeddings
from configs import load_config
from scripts.doc_parser import parse_documents


def sync_db():
    """
    Synchronize database with documents
    """
    conn = get_connection()
    config = load_config()
    doc_files = [
        Path(config["doc_parsing"]["input_dir"]) / doc
        for doc in config["doc_parsing"]["files"]
    ]
    faiss_index = get_faiss_index()
    recs_updated = 0
    recs_added = 0
    recs_deleted = 0
    with conn:
        for doc_path in doc_files:
            df_parsed = parse_documents(doc_path)
            section = df_parsed["section"][0]
            df_db = get_all_questions(conn, section)
            df_merged = pd.merge(
                df_parsed,
                df_db,
                on=["section", "subsection", "question"],
                how="outer",
                suffixes=("", "_db"),
            )
            df_deleted = df_merged[df_merged["answer"].isna()]  # deleted records
            df_added = df_merged[df_merged["answer_db"].isna()]  # new records
            df_updated = df_merged[
                df_merged["hash_answer"] != df_merged["hash_answer_db"]
            ].dropna()  # updated records
            if len(df_deleted):
                doc_ids = df_deleted["id"].tolist()
                delete_records(conn, doc_ids)
                faiss_index.remove_ids(np.array(doc_ids))
                recs_deleted += len(df_deleted)

            if len(df_added):
                embeddings = get_embeddings(df_added["text"].tolist())
                doc_ids = insert_records(conn, df_added)
                faiss_index.add_with_ids(embeddings, doc_ids)
                recs_added += len(df_added)

            if len(df_updated):
                doc_ids = df_updated["id"].tolist()
                embeddings = get_embeddings(df_updated["text"].tolist())
                update_records(conn, df_updated)
                faiss_index.remove_ids(np.array(doc_ids))
                faiss_index.add_with_ids(embeddings, doc_ids)
                recs_updated += len(df_updated)
    print(
        f"""DB synced
{recs_updated} records updated
{recs_added} records added
{recs_deleted} records delected
"""
    )
