import faiss
import os
from configs import load_config

faiss_index = None


def get_faiss_index():
    global faiss_index
    if faiss_index is None:
        config = load_config()
        if os.path.exists(config['faiss']['index_path']):
            faiss_index = faiss.read_index(config['faiss']['index_path'])
        else:
            faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(config['embedding']['dim']))
    return faiss_index


def save_index():
    faiss.write_index(faiss_index, load_config()['faiss']['index_path'])