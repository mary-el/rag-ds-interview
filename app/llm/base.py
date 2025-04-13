from abc import ABC, abstractmethod


class LLMInterface(ABC):

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass