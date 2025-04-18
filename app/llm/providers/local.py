from app.llm.base import LLMInterface


class LocalClient(LLMInterface):
    def __init__(self, base_url, model_name, **kwargs):
        self.url = f"{base_url}/v1/chat/completions"
        self.model = model_name
