"""
Offline: build a direction that suppresses the banned token’s log-prob.
Algorithm: Cosine-similarity gradient wrt target token embedding.
"""
import torch, json, tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import os

# Ensure the user has set their Hugging Face token
if "HUGGINGFACE_TOKEN" not in os.environ:
    raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable in your .env file")


TARGET = " orange"  # token with leading / trailing spaces for whole-word match
OUT = Path("vectors/no-citrus.pt")

tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mod = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tok_id = tok.encode(TARGET, add_special_tokens=False)[0]

emb = mod.get_input_embeddings().weight.data
direction = emb[tok_id]        # crude; improve with neighborhood averaging
direction = direction / torch.norm(direction)  # unit vector

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(direction, OUT)
print(f"Saved steer vector → {OUT}")
