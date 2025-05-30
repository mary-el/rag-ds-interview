import hashlib
from enum import Enum

import pandas as pd


class Mode(Enum):
    QA = "qa"
    QUIZ = "quiz"
    RATE = "rate"


def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_context(docs: pd.DataFrame) -> str:
    """
    Building context from documents
    :param docs: relevant docs dataframe
    :return: Context string
    """
    context = ""
    for i, doc in docs.iterrows():
        context += f"section: {doc['section']}\nsubsection: {doc['subsection']}\nquestion: {doc['question']}\nanswer: {doc['answer']}\n\n"
    return context
