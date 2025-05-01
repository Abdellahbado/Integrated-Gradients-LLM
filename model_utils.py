import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import os


def load_model(
    model_name: str, device: str = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Load a pretrained LLM and tokenizer."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    dtype = torch.float32
    if device == "cuda":
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",  
        low_cpu_mem_usage=True,
    ).eval()

    return model, tokenizer, device


def predict_next_token(
    model, tokenizer, text: str, device: str
) -> Tuple[str, float, int]:
    """
    Predict the most likely next token, its probability, and its ID for the given input text.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    next_token_id = torch.argmax(probs, dim=-1).item()

    top_values, top_indices = torch.topk(probs, 5)
    print("\nTop 5 token predictions:")
    for i, (idx, prob) in enumerate(
        zip(top_indices[0].tolist(), top_values[0].tolist())
    ):
        token_str = tokenizer.decode([idx])
        print(f"  {i+1}. ID: {idx}, Token: '{token_str}', Probability: {prob:.4f}")

    next_token_str = tokenizer.decode([next_token_id], skip_special_tokens=True)
    probability = probs[0, next_token_id].item()

    print(f"Selected token ID {next_token_id} -> '{next_token_str}'")

    return next_token_str, probability, next_token_id
