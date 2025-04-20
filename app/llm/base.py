from abc import ABC, abstractmethod


class LLMInterface(ABC):
    def __init__(
        self,
        model_name: str,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: str = None,
        do_sample: bool = False,
        num_beams: int = 1,
        top_p: float = 1.0,
        **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.top_p = top_p

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
