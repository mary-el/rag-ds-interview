import openai

from app.llm.base import LLMInterface
from configs import CONFIG


class OpenAIClient(LLMInterface):
    def __init__(self, model_name, **kwargs):
        openai.api_key = CONFIG['api_key']
        self.model_name = model_name

    def chat(self, messages):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages
        )
        return response['choices'][0]['message']['content']
