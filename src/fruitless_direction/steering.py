from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .prompts import SYSTEM as SYSTEM_PROMPT
from os import getenv

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
STEER_VECTOR_PATH = Path(
    getenv("STEER_VECTOR_PATH", "/modal/fruitless-direction/vectors/no-citrus.pt")
)

def load_model():
    tok = AutoTokenizer.from_pretrained(getenv("MODEL_ID", MODEL_ID))
    mod = AutoModelForCausalLM.from_pretrained(
        getenv("MODEL_ID", MODEL_ID), torch_dtype=torch.float16, device_map=getenv("DEVICE_MAP", "auto")
    )
    mod.eval()
    return pipeline("text-generation", model=mod, tokenizer=tok)

def apply_vector(embeds: torch.Tensor) -> torch.Tensor:
    steer = torch.load(STEER_VECTOR_PATH, map_location="cpu")
    return embeds - steer  # subtract “citrus” direction

def safe_generate(pipe, user_prompt: str, max_new_tokens: int = 512):
    prompt = f"{SYSTEM_PROMPT}\n\nUSER: {user_prompt}\nASSISTANT:"
    return pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
