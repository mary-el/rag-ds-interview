from abc import ABC, abstractmethod


class LLMInterface(ABC):
    def __init__(
        self,
        model_name,
        base_url=None,
        temperature=0.7,
        max_tokens=512,
        api_key=None,
        **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
