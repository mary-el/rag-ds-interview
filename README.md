# Data Sience Assistant

Welcome to my GitHub repository where I've developed a system that uses Large Language Models (LLMs) for data science learning and quizzing. My project aims at making complex topics more digestible through interactive dialogue.
I was preparing for a data scientist interview, so I've collected a lot of materials to study, but it was hard to navigate through it. I am intersted in LLMs, so I've created a Python project in which I use Retrieval Augmented Generation to help me. 

---

## Overview
DS Assistant is a lightweight system for helping users prepare for Data Science interviews.  
It supports two main modes:
- **RAG (Retrieval-Augmented Generation):** Answer user queries based on a database of materials.
- **Quiz Mode:** Generate open-ended questions from selected sections of the database for practice. Give your answer for model to rate or ask it to answer itself.

The project includes a CLI interface and a Telegram bot.

---

## Features
- Contextual answering using local knowledge base
- Local LLM deploy and OpenAI API support
- Interactive quiz generation
- Database synchronization from documents
- Fast FAISS-based search
- Telegram bot integration for remote usage

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mary-el/rag-ds-interview.git
   cd rag-ds-interview
   ```

2. Create and activate a virtual environment:
   ```bash
   conda create -n ds-assistant python=3.10
   conda activate ds-assistant
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:
   ```
   DB_HOST=you_database_host
   DB_PORT=db_port
   DB_NAME=db_name
   DB_USER=db_user_name
   DB_PASSWORD=db_password
   TELEGRAM_BOT_TOKEN=your_telegram_token
   API_KEY=key_for_LLM_api
   ```
5. Set configuration in `configs/config.yaml`
---

## Running

```bash
python main.py [-h] [--sync] [--config CONFIG] [--bot]

options:
  -h, --help            show this help message and exit
  --sync, -s            synchronize db with docs
  --config CONFIG, -c CONFIG
                        config file
  --bot, -b             run telegram bot
```

---

## Project Structure
```
app/
├── llm/             # LLM providers
bot/                 # Telegram bot
scripts/             # Utilities for data loading and database sync
configs              # Config files
data                 # Knowledge base documents
init                 # DB init file
main.py              # CLI entry point
```

---

## Requirements
- Python 3.10+
- FAISS
- OpenAI API (if external LLMs are used)
- PyTorch or Huggingface Transformers

---


## Data Sections
- [Classical ML](https://github.com/mary-el/rag-ds-interview/blob/main/docs/Classical_models/Classical_models.md)
- [Data](https://github.com/mary-el/rag-ds-interview/tree/main/docs/Data/Data.md)
- [Metrics](https://github.com/mary-el/rag-ds-interview/tree/main/docs/Metrics/Metrics.md)
- [Probability and Statistics](https://github.com/mary-el/rag-ds-interview/tree/main/docs/Probability_and_Statistics/Probability_and_Statistics.md)
- [Deep Learning](https://github.com/mary-el/rag-ds-interview/tree/main/docs/Deep%20Learning/Deep%20Learning.md)
- [Classical NLP](https://github.com/mary-el/rag-ds-interview/tree/main/docs/Classical_NLP/Classical_NLP.md)
- [LLM](https://github.com/mary-el/rag-ds-interview/tree/main/docs/LLM/LLM.md)

---



## Acknowledgments
I would like to thank for the courses on LLMs [DeepSchool](https://deepschool.ru/llm) and [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp)
