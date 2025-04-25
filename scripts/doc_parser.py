import os

import pandas as pd
import tqdm
from docx import Document

from app.logger import setup_logger
from app.utils import hash_text

logger = setup_logger(__name__)


def parse_documents(file: str) -> pd.DataFrame:
    subsections = []
    questions = []
    answers = []
    section = os.path.basename(file)[:-5].replace("_", " ")  # name of the section
    logger.info(f"Parsing Section {section}")
    doc = Document(file)
    current_answer = ""
    current_subsection = None

    for par in tqdm.tqdm(doc.paragraphs):
        if par.text.strip() == "":
            continue
        if par.style.name == "Heading 1":  # subsection started
            current_subsection = par.text
        elif par.style.name == "Heading 2":  # question started
            if current_answer:
                answers.append(current_answer)
            questions.append(par.text)
            subsections.append(current_subsection)
            current_answer = ""
        else:
            current_answer += par.text + "\n"
    answers.append(current_answer)
    qa_dataset = pd.DataFrame(
        {
            "section": [
                section,
            ]
            * len(subsections),
            "subsection": subsections,
            "question": questions,
            "answer": answers,
        }
    )
    qa_dataset["text"] = qa_dataset[
        ["section", "subsection", "question", "answer"]
    ].agg("\n".join, axis=1)
    qa_dataset["hash_answer"] = qa_dataset["answer"].map(hash_text)
    return qa_dataset
