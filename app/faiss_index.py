import os

import faiss

from configs import load_config

faiss_index = None
config = load_config()["faiss"]


def create_faiss_index() -> faiss.Index:
    global faiss_index
    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(config["dim"]))
    return faiss_index


def get_faiss_index() -> faiss.Index:
    global faiss_index
    os.makedirs("indexes", exist_ok=True)
    if faiss_index is None:
        if os.path.exists("indexes/" + config["index_path"]):
            faiss_index = faiss.read_index("indexes/" + config["index_path"])
        else:
            faiss_index = create_faiss_index()
    return faiss_index


def save_index():
    faiss_index = get_faiss_index()
    faiss.write_index(faiss_index, "indexes/" + config["index_path"])


def search_index(query_embedding, top_k=None):
    """
    Search FAISS index for similar documents
    
    :param query_embedding: Query embedding vector
    :param top_k: Number of documents to retrieve (defaults to config records_num)
    :return: Tuple of (document_ids, scores)
    """
    if top_k is None:
        top_k = config["records_num"]
    faiss_index = get_faiss_index()
    scores, ids = faiss_index.search(query_embedding, top_k)
    return ids[0].tolist(), scores[0].tolist()


def add_with_ids(embeddings, doc_ids):
    faiss_index = get_faiss_index()
    faiss_index.add_with_ids(embeddings, doc_ids)
