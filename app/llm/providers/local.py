from openai import api_key
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from app.llm.base import LLMInterface
from app.logger import setup_logger

logger = setup_logger(__name__)


def move_to_device(d, device):
    for k, v in d.items():
        d[k] = v.to(device)
    return d


class LocalClient(LLMInterface):
    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map=device, token=self.api_key
        ).to(device)
        logger.info(f"Loaded weights for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, device=device, token=self.api_key
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gc = GenerationConfig(
            do_sample=self.do_sample,
            temperature=self.temperature,
            num_beams=self.num_beams,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.pad_token_type_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model.generation_config = self.gc

    def generate(self, prompt: str, **kwargs) -> str:
        input_tokens = move_to_device(
            self.tokenizer(prompt, return_tensors="pt"),
            self.device,
        )
        generated = self.model.generate(**input_tokens).cpu().detach()
        generated = generated[0, len(input_tokens[0]) :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer
