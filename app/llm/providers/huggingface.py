from huggingface_hub import InferenceClient

from app.llm.base import LLMInterface


class HuggingFaceClient(LLMInterface):
    def __init__(self, model_name, api_key, **kwargs):
        self.model_name = model_name
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.client = InferenceClient(
            # provider="hf-inference",
            api_key=api_key,
        )

    def generate(self, prompt: str) -> str:
        return self.client.text_generation(prompt=prompt, model=self.model_name)
