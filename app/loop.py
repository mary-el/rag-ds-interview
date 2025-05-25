from app.database import get_all_sections
from app.logger import setup_logger
from app.quiz import quiz_pipeline, rate_answer_pipeline
from app.rag import rag_pipeline
from scripts import load_records_to_db_and_faiss, sync_db


def run_loop(synchronize: float = False):
    logger = setup_logger("main")
    logger.info("APP STARTED")

    if load_records_to_db_and_faiss() and synchronize:
        sync_db()

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

        if query.lower() not in {"quiz", "question"}:  # user asked a question
            answer = rag_pipeline(query)
            print(f"\n💡 Answer:\n{answer}")
            logger.info(f"MODEL ANSWER: {answer}")
        else:  # quiz mode
            logger.info("QUIZ MODE")

            sections = get_all_sections()
            quit_quiz_mode_opt = len(sections) + 1
            section_list = (
                " ".join((f"{i}. {section}" for i, section in enumerate(sections, 1)))
                + f" {quit_quiz_mode_opt}. Quit quiz mode"
            )

            while True:
                print(f"\n🌈 Available quiz sections: {section_list}\n")
                section_n = int(input("Write a number: "))

                if section_n == quit_quiz_mode_opt:
                    break

                if not 1 <= section_n <= len(sections):
                    print("Wrong number")
                    continue

                section = sections[section_n - 1]
                logger.info(f"SECTION {section}")
                quiz_question, context = quiz_pipeline(section)

                print(f"\n❓ Question:\n{quiz_question}")
                logger.info(f"MODEL QUESTION: {quiz_question}")
                query = input(
                    f'Answer the question for me to rate it or write "answer" if you want to learn it\n'
                ).strip()

                if query.lower() in {"answer"}:  # user wants to know the answer
                    answer = rag_pipeline(quiz_question, context)
                    print(f"\n💡 Answer:\n{answer}")
                    logger.info(f"MODEL ANSWER: {answer}")
                else:  # user gives their answer
                    evaluation = rate_answer_pipeline(
                        context=context, question=quiz_question, answer=query
                    )
                    print(f"\n👍 {evaluation}")
                    logger.info(f"ANSWER EVALUATION: {evaluation}")
