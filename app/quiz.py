from app.database import get_random_questions
from app.generator import generate_text
from app.utils import Mode, get_context


def quiz_pipeline(section: str = None):
    # Getting random question from the chosen section
    documents = get_random_questions(section)
    if len(documents) == 0:
        return "The section is empty"
    # Getting context from the questions
    context = get_context(documents)
    # Getting question from LLM
    question = generate_text({"context": context}, Mode.QUIZ)

    return question
