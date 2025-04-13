from dotenv import load_dotenv
import os

load_dotenv()

CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "api_key": os.getenv("API_KEY")  # API key for the LLM provider
}
