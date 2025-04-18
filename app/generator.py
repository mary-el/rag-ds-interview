from pathlib import Path
from typing import Dict, Optional

from app.llm import get_llm_client
from app.utils import Mode
from configs import load_config

config = load_config()
llm = get_llm_client(config)

templates: Optional[Dict] = None


def get_templates(root: Path = Path("configs/templates")) -> Dict[Mode, str]:
    global templates

    if templates is None:
        templates = {}
        for mode in Mode:
            file = root / config["llm"]["template_files"][mode.value]
            with open(file, "r") as f:
                templates[mode] = f.read()
    return templates


def build_prompt(query: str, context: str, mode: Mode):

    prompt_template = get_templates()[mode].strip()
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def generate_answer(question: str, context: str) -> str:
    """
    Work in progress
    :param question: User question
    :param context: Context string
    :return: Model answer
    """
    prompt = build_prompt(question, context, mode=Mode.QA)
    result = llm.generate(prompt)
    return result
