import os

import faiss

from configs import load_config

faiss_index = None
config = load_config()


def create_faiss_index() -> faiss.Index:
    global faiss_index
    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(config["faiss"]["dim"]))
    return faiss_index


def get_faiss_index() -> faiss.Index:
    global faiss_index
    if faiss_index is None:
        if os.path.exists(config["faiss"]["index_path"]):
            faiss_index = faiss.read_index(config["faiss"]["index_path"])
        else:
            faiss_index = create_faiss_index()
    return faiss_index


def save_index():
    faiss_index = get_faiss_index()
    faiss.write_index(faiss_index, load_config()["faiss"]["index_path"])


def search_index(query_embedding, top_k: int):
    faiss_index = get_faiss_index()
    scores, ids = faiss_index.search(query_embedding, top_k)
    return ids[0].tolist(), scores[0].tolist()


def add_with_ids(embeddings, doc_ids):
    faiss_index = get_faiss_index()
    faiss_index.add_with_ids(embeddings, doc_ids)
