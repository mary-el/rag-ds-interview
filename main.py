import os

from app.rag import rag_pipeline
from scripts import load_records_to_db_and_faiss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    load_records_to_db_and_faiss()

    while True:
        query = input("\nEnter your question: ")
        if query.lower() in {"exit", "quit"}:
            break

        answer = rag_pipeline(query)
        print(f"\n💡 Answer:\n{answer}")
