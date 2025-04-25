from transformers import AutoModelForCausalLM, AutoTokenizer

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
            self.model_name, torch_dtype="auto", device_map=device
        ).to(device)
        logger.info(f"Loaded weights for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, device=device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, **kwargs) -> str:
        input_tokens = move_to_device(
            self.tokenizer(
                prompt, return_tensors="pt", max_length=self.max_tokens, truncation=True
            ),
            self.device,
        )
        generated = (
            self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                num_beams=self.num_beams,
                top_p=self.top_p,
            )
            .cpu()
            .detach()
        )
        generated = generated[0, len(input_tokens[0]) :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        return answer
