import openai

from app.database import get_random_questions
from app.generator import generate_text
from app.logger import setup_logger
from app.utils import Mode, get_context

logger = setup_logger(__name__)


def quiz_pipeline(section: str = None) -> dict:
    try:
        # Getting random question from the chosen section
        documents = get_random_questions(section)
        if len(documents) == 0:
            return {"success": False, "error": "The section is empty"}
        # Getting context from the questions
        context = get_context(documents)
        # Getting question from LLM
        question = generate_text({"context": context}, Mode.QUIZ)
        return {"success": True, "question": question, "context": context}
    except openai.APITimeoutError as e:
        logger.error(f"Timeout occurred: {e}")
        return {"success": False, "error": "Timeout occurred"}
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        return {"success": False, "error": e}


def rate_answer_pipeline(context: str, question: str, answer: str) -> dict:
    try:  # rating user's answer
        evaluation = generate_text(
            {"context": context, "question": question, "answer": answer}, mode=Mode.RATE
        )
        return {"success": True, "evaluation": evaluation}
    except openai.APITimeoutError as e:
        logger.error(f"Timeout occurred: {e}")
        return {"success": False, "error": "Timeout occurred"}
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        return {"success": False, "error": e}
