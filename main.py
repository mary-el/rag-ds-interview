import argparse
import os

from app.database import get_all_sections
from app.quiz import quiz_pipeline
from app.rag import rag_pipeline
from scripts import load_records_to_db_and_faiss, sync_db

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync", "-s", help="synchronize db with docs", action="store_true"
    )
    args = parser.parse_args()

    if load_records_to_db_and_faiss() and args.sync:
        sync_db()

    while True:
        query = input("\nEnter your question: ").strip()

        if len(query) == 0:
            continue

        if query.lower() in {"exit", "quit"}:
            break

        if query.lower() in {"quiz", "question"}:  # quiz mode
            sections = get_all_sections()
            section_list = " ".join(
                (f"{i}. {section}" for i, section in enumerate(sections, 1))
            )
            print(f"\nAvailable sections: {section_list}\n")
            section_n = int(input("Write a number: "))
            quiz_question = quiz_pipeline(sections[section_n - 1])
            print(f"\n  Question:\n{quiz_question}")
            continue

        answer = rag_pipeline(query)
        print(f"\n💡 Answer:\n{answer}")
