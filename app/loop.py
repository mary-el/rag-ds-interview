from app.database import get_all_sections
from app.logger import setup_logger
from app.quiz import quiz_pipeline, rate_answer_pipeline
from app.rag import rag_pipeline
from scripts import load_records_to_db_and_faiss, sync_db


def display_sections(sections):
    section_lines = [f"{i}. {section}" for i, section in enumerate(sections, 1)]
    section_lines.append(f"{len(sections) + 1}. Quit quiz mode")
    print("\n🌈 Available quiz sections:")
    print("\n".join(section_lines))


def handle_quiz(sections, logger) -> bool:
    display_sections(sections)

    try:
        section_n = int(input("\nWrite a number: "))
    except ValueError:
        print("❌ Please enter a valid number.")
        return True

    if section_n == len(sections) + 1:
        return False

    if not 1 <= section_n <= len(sections):
        print("❌ Wrong number")
        return True

    section = sections[section_n - 1]
    logger.info(f"SECTION: {section}")

    response = quiz_pipeline(section)
    if not response.get("success"):
        print(f"💥 {response['error']}")
        return True

    question = response["question"]
    context = response["context"]

    print(f"\n❓ Question:\n{question}")
    logger.info(f"MODEL QUESTION: {question}")

    user_input = input(
        'Answer the question or type "answer" to see the correct one:\n'
    ).strip()

    if user_input.lower() == "answer":
        response = rag_pipeline(question, context)
        if not response.get("success"):
            print(f"💥 {response['error']}")
            return True
        answer = response["answer"]
        print(f"\n💡 Answer:\n{answer}")
        logger.info(f"MODEL ANSWER: {answer}")
    else:
        response = rate_answer_pipeline(
            context=context, question=question, answer=user_input
        )
        if not response.get("success"):
            print(f"💥 {response['error']}")
            return True
        evaluation = response["evaluation"]
        print(f"\n👍 {evaluation}")
        logger.info(f"ANSWER EVALUATION: {evaluation}")

    return True


def handle_user_question(query: str, logger):
    response = rag_pipeline(query)
    if response.get("success"):
        answer = response["answer"]
        print(f"\n💡 Answer:\n{answer}")
        logger.info(f"MODEL ANSWER: {answer}")
    else:
        print(f"💥 {response['error']}")


def run_loop(synchronize: bool = False):
    logger = setup_logger("main")
    logger.info("APP STARTED")

    if load_records_to_db_and_faiss() and synchronize:
        sync_db()

    print("Write 'quiz' to enter quiz mode")
    logger.info("ENTERING MAIN LOOP")

    while True:
        query = input("\nEnter your question: ").strip()
        if not query:
            continue

        logger.info(f"USER'S QUERY: {query}")

        if query.lower() in {"exit", "quit"}:
            logger.info("EXITING")
            break

        if query.lower() in {"quiz", "question"}:
            logger.info("QUIZ MODE")
            sections = get_all_sections()
            while handle_quiz(sections, logger):
                pass
        else:
            handle_user_question(query, logger)
