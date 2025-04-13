import requests
from huggingface_hub import InferenceClient

from app.llm.base import LLMInterface
from configs.env import CONFIG


class HuggingFaceClient(LLMInterface):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.headers = {"Authorization": f"Bearer {CONFIG['api_key']}"}
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.client = InferenceClient(
            # provider="hf-inference",
            api_key=CONFIG['api_key'],
        )

    def chat(self, messages):
        prompt = "\n".join([m["content"] for m in messages])
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt})
        return response.json()[0]["generated_text"]

    def generate(self, prompt: str) -> str:
        return self.client.text_generation(prompt=prompt, model=self.model_name)
