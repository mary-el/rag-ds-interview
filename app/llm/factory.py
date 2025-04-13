from app.llm.providers import LocalClient, OpenAIClient, HuggingFaceClient
from app.llm.base import LLMInterface


def get_llm_client(config) -> LLMInterface:
    provider = config["llm"]["provider"]
    params = config["llm"]

    if provider == "openai":
        return OpenAIClient(**params)
    elif provider == "huggingface":
        return HuggingFaceClient(**params)
    elif provider == "local":
        return LocalClient(**params)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
