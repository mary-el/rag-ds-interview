from app.llm.base import LLMInterface
from app.llm.providers import LocalClient, OpenAICompatibleClient


def get_llm_client(config) -> LLMInterface:
    provider = config["llm"]["provider"]
    params = config["llm"]

    if provider == "openai":
        return OpenAICompatibleClient(**params)
    elif provider == "local":
        return LocalClient(**params)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
