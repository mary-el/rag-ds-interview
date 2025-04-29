import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from configs import load_config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_embedding_model(model_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_ckpt, trust_remote_code=True)
    return model, tokenizer


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


@torch.no_grad()
def get_embeddings(text_list, batch_size=64):
    config = load_config()
    model, tokenizer = get_embedding_model(config["faiss"]["model"])
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i : i + batch_size]
        encoded_input = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        embeddings.append(cls_pooling(model_output).detach().numpy())
    return np.concatenate(embeddings)
