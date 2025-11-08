#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 1: Layer-wise analysis for P(T) only (Target-language passage)

- Loads a Belebele-style CSV
- Builds P(T) prompts and runs them through a LLaMA-style model
- For each layer:
    * ProtoAcc_T            : logit-lens proto accuracy vs gold
    * ConsistentWithFinal_T : agreement with final-layer prediction
    * MeanLogP_margin_T     : logp(correct) - max logp(other options)
    * Silhouette_lang_T     : PCA-based silhouette over dialects
    * BCSS_WCSS_lang_T      : between/within cluster ratio for dialects
    * Entropy_lang_T        : language entropy
    * ExplainedVar_PC1_T / PC2_T : PCA explained variance

- Saves:
    layer_metrics_T.csv
    pca_points_T.csv
"""

import os
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# -----------------------
# Prompt templates
# -----------------------

EVAL_T = """Read the passage thoroughly and answer the question based on the information provided.
Lets think step by step.
Carefully evaluate all the choices and eliminate the incorrect options.
Select the single best-supported answer as your response.

Passage:
{flores_passage}

Question:
{question}

Choices:
1: {mc_answer1}
2: {mc_answer2}
3: {mc_answer3}
4: {mc_answer4}

Strictly follow this format: "[[1]]" or "[[2]]" or "[[3]]" or "[[4]]". Do not provide any feedback.

Your Answer:
"""

SYSTEM_MSG = (
    "You are a multilingual MCQA model. "
    "Lets think step by step. First, explain your reasoning. "
    "Then clearly state your final answer using [[1]], [[2]], [[3]], or [[4]]."
)


def build_prompt_T(row: pd.Series) -> str:
    return EVAL_T.format(**row.to_dict())


def build_chat_input(prompt: str) -> str:
    # Same pattern you used before
    return f"<|system|>\n{SYSTEM_MSG}\n<|user|>\n{prompt}\n<|assistant|>\n"


# -----------------------
# Helper functions
# -----------------------

@torch.no_grad()
def mean_pool_hidden(hidden_states, attn_mask):
    """
    hidden_states: list[L+1] of (B,T,D)
    attn_mask    : (B,T)
    Returns dict[layer -> (B,D)] mean-pooled over tokens (masked).
    """
    pooled = {}
    mask = attn_mask.unsqueeze(-1)  # (B,T,1)
    for layer_idx, h in enumerate(hidden_states):
        denom = mask.sum(dim=1).clamp(min=1)
        m = (h * mask).sum(dim=1) / denom
        pooled[layer_idx] = m.detach().cpu().numpy()
    return pooled


@torch.no_grad()
def collect_T_hidden_and_logits(
    df_chunk: pd.DataFrame,
    tokenizer,
    model,
    lm_W,
    lm_b,
    digit_ids,
    device: str,
    batch_size: int = 8,
):
    """
    For P(T) prompts:
      - get mean-pooled hidden states per layer (for PCA)
      - logit-lens over digits 1..4 at EACH layer
      - final-layer prediction over digits (P(T)_final)

    Returns:
      pooled_by_layer: dict[layer] -> np.array[N, D]  (mean-pooled)
      layer_scores_T : dict[layer] with keys:
                          pred_digit, correct_digit,
                          logp_correct, logp_margin
      final_pred_T   : np.array[N]   (final-layer predicted digits)
      gold_T         : np.array[N]   (gold labels)
    """
    prompts = [build_chat_input(build_prompt_T(r)) for _, r in df_chunk.iterrows()]
    gold_T = df_chunk["correct_answer_num"].astype(int).values

    pooled_by_layer = defaultdict(list)
    layer_scores_T = defaultdict(lambda: defaultdict(list))
    final_pred_T_all = []

    print(f"[Module1] Collecting hidden states + logits for {len(prompts)} examples (P(T))")

    for start in tqdm(range(0, len(prompts), batch_size), desc="[Module1] Forward batches"):
        batch_prompts = prompts[start:start + batch_size]
        batch_gold = gold_T[start:start + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        out = model(
            **enc,
            output_hidden_states=True,
            use_cache=False
        )
        hidden_states = out.hidden_states    # list[L+1] of (B,T,D)
        attn_mask = enc["attention_mask"]    # (B,T)

        B, T = attn_mask.shape
        last_pos = attn_mask.sum(dim=1) - 1  # (B,) index of last non-pad token

        # ---- mean-pooled for PCA
        pooled = mean_pool_hidden(hidden_states, attn_mask)
        for l, vec in pooled.items():
            pooled_by_layer[l].append(vec)  # each is (B,D)

        # ---- logit-lens for digits per layer
        num_layers = len(hidden_states)

        for layer_idx in range(num_layers):
            h = hidden_states[layer_idx]  # (B,T,D)

            # gather h at last_pos
            idx = last_pos.view(B, 1, 1).expand(-1, 1, h.size(-1))  # (B,1,D)
            h_last = torch.gather(h, dim=1, index=idx).squeeze(1)   # (B,D)

            # logits from this layer
            logits = torch.matmul(h_last, lm_W.T) + lm_b  # (B,V)

            digit_logits = torch.stack([
                logits[:, digit_ids[1]],
                logits[:, digit_ids[2]],
                logits[:, digit_ids[3]],
                logits[:, digit_ids[4]],
            ], dim=-1)  # (B,4)

            log_probs = torch.log_softmax(digit_logits.float(), dim=-1)  # (B,4)

            # predicted digit at this layer
            pred_idx = torch.argmax(log_probs, dim=-1)   # 0..3
            pred_digit = (pred_idx + 1).cpu().numpy()    # 1..4

            # logp(correct), logp_margin
            gold_idx = (torch.tensor(batch_gold, device=log_probs.device) - 1).clamp(0, 3)
            logp_correct = log_probs[torch.arange(B, device=log_probs.device), gold_idx]

            mask = torch.ones_like(log_probs, dtype=torch.bool)
            mask[torch.arange(B), gold_idx] = False
            other_logp = log_probs.masked_fill(~mask, -1e9)
            max_other = other_logp.max(dim=-1).values
            logp_margin = logp_correct - max_other

            # store
            layer_scores_T[layer_idx]["pred_digit"].extend(pred_digit.tolist())
            layer_scores_T[layer_idx]["correct_digit"].extend(batch_gold.tolist())
            layer_scores_T[layer_idx]["logp_correct"].extend(logp_correct.cpu().tolist())
            layer_scores_T[layer_idx]["logp_margin"].extend(logp_margin.cpu().tolist())

            if layer_idx == num_layers - 1:
                # final layer prediction used as "P(T)_final"
                final_pred_T_all.extend(pred_digit.tolist())

    # stack pooled
    for l in pooled_by_layer.keys():
        pooled_by_layer[l] = np.vstack(pooled_by_layer[l])  # (N,D)

    # finalize layer_scores arrays
    for l, dd in layer_scores_T.items():
        for k in dd.keys():
            layer_scores_T[l][k] = np.array(dd[k])

    final_pred_T = np.array(final_pred_T_all)
    return pooled_by_layer, layer_scores_T, final_pred_T, gold_T


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Module 1: Layer-wise P(T) analysis")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Belebele-style CSV with flores_passage, question, mc_answer1..4, correct_answer_num, dialect, unique_key")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for layer_metrics_T.csv and pca_points_T.csv")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HF model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device, e.g., cuda:0")
    parser.add_argument("--max_samples", type=int, default=400,
                        help="Max number of rows to use for this module (subsample if larger)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for forward passes")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Torch dtype for model")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load data ----------
    print(f"[Module1] Loading data from: {args.data_csv}")
    df = pd.read_csv(args.data_csv)

    needed = [
        "unique_key", "dialect",
        "flores_passage", "question",
        "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4",
        "correct_answer_num"
    ]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column in CSV: {c}")

    df = df[df["correct_answer_num"].isin([1, 2, 3, 4])].reset_index(drop=True)
    print(f"[Module1] Total rows with valid correct_answer_num ∈ {{1..4}}: {len(df)}")

    # Subsample for this module
    rng = np.random.RandomState(42)
    if len(df) > args.max_samples:
        idx = rng.choice(df.index.values, size=args.max_samples, replace=False)
        df_T = df.loc[idx].reset_index(drop=True)
        print(f"[Module1] Subsampled to {len(df_T)} rows (max_samples={args.max_samples})")
    else:
        df_T = df.copy()
        print(f"[Module1] Using all {len(df_T)} rows (<= max_samples)")

    # ---------- Hugging Face login (optional) ----------
    # Expect HF token in HF_TOKEN or HUGGING_FACE_HUB_TOKEN, OR you already did `huggingface-cli login`.
    from huggingface_hub import login

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("[Module1] Logging into Hugging Face Hub using HF_TOKEN / HUGGING_FACE_HUB_TOKEN")
        login(token=hf_token)
    else:
        print("[Module1] No HF token found in env; assuming you already ran `huggingface-cli login` or have the model cached.")

    # ---------- Load model ----------
    print(f"[Module1] Loading model from Hugging Face: {args.model_name}")
    print(f"[Module1] Device: {args.device}, dtype: {args.dtype}")

    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Put the full 8B model on a single GPU (A40 has enough VRAM)
    # This keeps all hidden states on one device, which makes logit-lens math clean.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map={"": args.device},   # single GPU; change to "auto" later if we really need sharding
    )
    model.eval()

    lm_head = model.lm_head
    lm_W = lm_head.weight
    lm_b = lm_head.bias

    # Get token IDs for digits 1–4
    print("[Module1] Resolving digit token IDs for labels 1,2,3,4")
    digit_ids = {}
    with torch.no_grad():
        for k in [1, 2, 3, 4]:
            ids = tokenizer(str(k), add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                print(f"[Module1][WARN] Label {k} is not a single token: {ids}")
            digit_ids[k] = ids[0]
    print(f"[Module1] Digit IDs: {digit_ids}")

    # ---------- Collect hidden states + logits ----------
    pooled_T, layer_scores_T, final_pred_T, gold_T = collect_T_hidden_and_logits(
        df_T,
        tokenizer,
        model,
        lm_W,
        lm_b,
        digit_ids,
        args.device,
        batch_size=args.batch_size,
    )

    num_layers = len(layer_scores_T)
    print(f"[Module1] Collected hidden/logits for {num_layers} layers (including embedding layer 0)")

    # ---------- Per-layer proto accuracy + consistency ----------
    print("[Module1] Computing per-layer proto accuracy + consistency vs final layer")
    rows_T = []
    for l in sorted(layer_scores_T.keys()):
        sc = layer_scores_T[l]
        pred = sc["pred_digit"]
        gold = sc["correct_digit"]

        proto_acc = (pred == gold).mean()
        same_as_final = (pred == final_pred_T).mean()
        mean_margin = float(np.mean(sc["logp_margin"]))

        rows_T.append({
            "layer": l,
            "ProtoAcc_T": proto_acc,
            "ConsistentWithFinal_T": same_as_final,
            "MeanLogP_margin_T": mean_margin,
        })

    layer_metrics_T = pd.DataFrame(rows_T).sort_values("layer").reset_index(drop=True)

    # ---------- PCA + geometry per layer ----------
    print("[Module1] Running PCA + geometry metrics per layer")
    dialects = df_T["dialect"].tolist()
    N_T = len(df_T)

    pca_rows_T = []
    geom_rows_T = []

    for l in sorted(pooled_T.keys()):
        X = pooled_T[l]  # (N_T, D)

        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)

        sil = silhouette_score(X2, dialects)

        counts = Counter(dialects)
        probs = np.array(list(counts.values()), dtype=float)
        probs = probs / probs.sum()
        entropy = -(probs * np.log(probs + 1e-12)).sum()

        overall_mean = X2.mean(axis=0)
        langs = np.unique(dialects)
        bcss, wcss = 0.0, 0.0
        for lang in langs:
            idxs = [i for i, la in enumerate(dialects) if la == lang]
            pts = X2[idxs]
            center = pts.mean(axis=0)
            bcss += len(pts) * np.linalg.norm(center - overall_mean) ** 2
            wcss += ((pts - center) ** 2).sum()
        bcss_wcss = bcss / max(wcss, 1e-6)

        geom_rows_T.append({
            "layer": l,
            "Silhouette_lang_T": sil,
            "BCSS_WCSS_lang_T": bcss_wcss,
            "Entropy_lang_T": entropy,
            "ExplainedVar_PC1_T": float(pca.explained_variance_ratio_[0]),
            "ExplainedVar_PC2_T": float(pca.explained_variance_ratio_[1]),
        })

        for i in range(N_T):
            pca_rows_T.append({
                "layer": l,
                "pc1": X2[i, 0],
                "pc2": X2[i, 1],
                "dialect": dialects[i],
                "correct_final": int(gold_T[i] == final_pred_T[i]),
            })

        print(f"[Module1] Layer {l:02d}: Sil={sil:.3f}, Entropy={entropy:.3f}, BCSS/WCSS={bcss_wcss:.3f}")

    geom_T = pd.DataFrame(geom_rows_T).sort_values("layer").reset_index(drop=True)
    pca_T = pd.DataFrame(pca_rows_T)

    layer_metrics_T_full = layer_metrics_T.merge(geom_T, on="layer", how="left").sort_values("layer")

    out_metrics = os.path.join(args.out_dir, "layer_metrics_T.csv")
    out_pca = os.path.join(args.out_dir, "pca_points_T.csv")

    layer_metrics_T_full.to_csv(out_metrics, index=False)
    pca_T.to_csv(out_pca, index=False)

    print(f"[Module1] Saved layer-wise metrics to: {out_metrics}")
    print(f"[Module1] Saved PCA points to       : {out_pca}")
    print("[Module1] Done.")
if __name__ == "__main__":
    main()