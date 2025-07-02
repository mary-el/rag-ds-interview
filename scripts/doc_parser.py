import os

import pandas as pd
import tqdm
from docx import Document
from langchain_text_splitters import CharacterTextSplitter

from app.logger import setup_logger
from app.utils import hash_text
from configs import load_config

config = load_config()["doc_parsing"]
text_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"],
)

logger = setup_logger(__name__)


def add_new_answer(
    current_answer,
    current_question,
    current_subsection,
    answers,
    questions,
    subsections,
):
    chunks = text_splitter.split_text(current_answer)  # chunking long answers
    for i, chunk in enumerate(chunks):
        questions.append(current_question + "_" + str(i + 1))
        answers.append(chunk)
        subsections.append(current_subsection)


def parse_document(file: str) -> pd.DataFrame:
    subsections = []
    questions = []
    answers = []

    section = os.path.basename(file)[:-5].replace("_", " ")  # name of the section
    logger.info(f"Parsing Section {section}")
    doc = Document(file)

    current_answer = ""
    current_subsection = None
    current_question = ""

    for par in tqdm.tqdm(doc.paragraphs):
        if par.text.strip() == "":
            continue

        if par.style.name == "Heading 1":  # subsection started
            current_subsection = par.text

        elif par.style.name == "Heading 2":  # question started
            if current_answer:
                add_new_answer(
                    current_answer,
                    current_question,
                    current_subsection,
                    answers,
                    questions,
                    subsections,
                )
            current_question = par.text
            current_answer = ""
        else:
            current_answer += par.text + "\n"
    add_new_answer(
        current_answer,
        current_question,
        current_subsection,
        answers,
        questions,
        subsections,
    )

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
