import openai

from app.llm.base import LLMInterface


class OpenAICompatibleClient(LLMInterface):
    def generate(self, prompt: str, system_prompt=None):
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return response.choices[0].message.content
