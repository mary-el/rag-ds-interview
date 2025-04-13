import requests

from app.llm.base import LLMInterface


class LocalClient(LLMInterface):
    def __init__(self, base_url, model_name, **kwargs):
        self.url = f"{base_url}/v1/chat/completions"
        self.model = model_name

    def chat(self, messages):
        payload = {"model": self.model, "messages": messages}
        response = requests.post(self.url, json=payload)
        return response.json()['choices'][0]['message']['content']
