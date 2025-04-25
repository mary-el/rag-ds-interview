from app.database import get_random_questions
from app.generator import generate_text
from app.logger import setup_logger
from app.utils import Mode, get_context

logger = setup_logger(__name__)


def quiz_pipeline(section: str = None):
    try:
        # Getting random question from the chosen section
        documents = get_random_questions(section)
        if len(documents) == 0:
            return "The section is empty"
        # Getting context from the questions
        context = get_context(documents)
        # Getting question from LLM
        question = generate_text({"context": context}, Mode.QUIZ)
        return question, context

    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        return f"Exception occurred: {e}", None
