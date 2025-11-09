#!/usr/bin/env python
# test_llama_hello.py
#
# Minimal sanity test:
# - login to Hugging Face (if HF_TOKEN / HUGGING_FACE_HUB_TOKEN is set)
# - load meta-llama/Meta-Llama-3.1-8B-Instruct from HF
# - print a short generation

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


def main():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    print("=== test_llama_hello.py ===")
    print(f"Host: {os.uname().nodename}")
    print(f"Working dir: {os.getcwd()}")
    print(f"Model name: {model_name}")

    # ---------- HF login ----------
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("[HELLO] Found HF token in environment; logging into Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("[HELLO] No HF token env var found. Assuming you already ran `huggingface-cli login` or have the model cached.")

    # ---------- Load tokenizer + model ----------
    print("[HELLO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[HELLO] Loading model with device_map='auto' and torch.float16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()

    n_layers = getattr(model.config, "num_hidden_layers", "NA")
    hidden_sz = getattr(model.config, "hidden_size", "NA")
    print(f"[HELLO] Model loaded: num_layers={n_layers}, hidden_size={hidden_sz}")

    # ---------- Tiny generation ----------
    prompt = "Say hello from the LLaMA model in one short sentence."
    print("[HELLO] Running a tiny generation...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    print("\n[HELLO] Generation output:")
    print(out)
    print("\n[HELLO] Test complete. ✅")


if __name__ == "__main__":
    main()
