from app.database import get_all_sections
from app.logger import setup_logger
from app.quiz import quiz_pipeline
from app.rag import rag_pipeline
from scripts import load_records_to_db_and_faiss, sync_db


def run_loop(synchronize: float = False):
    logger = setup_logger("main")
    logger.info("APP STARTED")

    if load_records_to_db_and_faiss() and synchronize:
        sync_db()

    quiz_question = None
    current_context = None

    print("Write 'quiz' to enter quiz mode")
    logger.info("ENTERING MAIN LOOP")

    while True:
        query = input("\nEnter your question: ").strip()

        if len(query) == 0:
            continue
        logger.info(f"USER'S QUERY: {query}")

        if query.lower() in {"exit", "quit"}:
            logger.info("EXITING")
            break

        if query.lower() in {"quiz", "question"}:  # quiz mode
            logger.info("QUIZ MODE")
            sections = get_all_sections()
            section_list = " ".join(
                (f"{i}. {section}" for i, section in enumerate(sections, 1))
            )
            print(f"\nAvailable sections: {section_list}\n")
            section_n = int(input("Write a number: "))
            if not 1 <= section_n <= len(sections):
                print("Wrong number")
                continue
            section = sections[section_n - 1]
            logger.info(f"SECTION {section}")
            quiz_question, context = quiz_pipeline(section)

            print(f"\n❓ Question:\n{quiz_question}")
            print(f'Write "answer" if you want to know the answer\n')
            logger.info(f"MODEL QUESTION: {quiz_question}")

            continue

        if query.lower() in {"answer"} and quiz_question:
            query = quiz_question
            current_context = context

        answer = rag_pipeline(query, current_context)
        print(f"\n💡 Answer:\n{answer}")
        logger.info(f"MODEL ANSWER: {answer}")
        quiz_question = None
        current_context = None
