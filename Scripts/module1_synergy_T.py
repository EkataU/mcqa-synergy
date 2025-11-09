#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 1 (P(T) only): Layer-wise analysis for Belebele-style MCQA

What it computes per LLaMA layer (including layer 0):
  - ProtoAcc_T (logit-lens proto accuracy vs gold)
  - ConsistentWithFinal_T (agreement with final-layer prediction)
  - MeanLogP_margin_T (log p(correct) - max log p(other))
  - MeanPredEntropy_T (predictive entropy)  <-- Beyond-the-Final-Layer style
  - ECE_T (Expected Calibration Error)      <-- Beyond-the-Final-Layer style
  - Silhouette_lang_T, BCSS_WCSS_lang_T (PCA geometry over dialects)

Also writes per-dialect metrics and a per-example file (per layer) to enable
cross-run consistency aggregation.

Outputs (under out_dir/{run_tag}_T/):
  - layer_metrics_T.csv           (overall + per-dialect rows)
  - pca_points_T.csv              (2D PCA points per layer for plotting)
  - per_example_T.csv             (row_id-level preds/probs/entropy per layer)
"""

import os
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
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
    # Match earlier chat pattern
    return f"<|system|>\n{SYSTEM_MSG}\n<|user|>\n{prompt}\n<|assistant|>\n"


# -----------------------
# Helpers
# -----------------------

@torch.no_grad()
def mean_pool_hidden(hidden_states, attn_mask):
    """
    hidden_states: list[L+1] of (B,T,D)
    attn_mask    : (B,T)
    -> dict[layer -> (B,D)] mean-pooled token embeddings (masked)
    """
    pooled = {}
    mask = attn_mask.unsqueeze(-1)  # (B,T,1)
    for layer_idx, h in enumerate(hidden_states):
        denom = mask.sum(dim=1).clamp(min=1)
        m = (h * mask).sum(dim=1) / denom
        pooled[layer_idx] = m.detach().cpu().numpy()
    return pooled


def compute_ece(confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error over n_bins bins.
    confidences: [N] float in [0,1]
    corrects   : [N] 0/1
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            idx = (confidences >= lo) & (confidences < hi)
        else:
            idx = (confidences >= lo) & (confidences <= hi)
        if idx.sum() == 0:
            continue
        acc_bin = corrects[idx].mean()
        conf_bin = confidences[idx].mean()
        ece += (idx.mean()) * abs(acc_bin - conf_bin)
    return float(ece)


@torch.no_grad()
def collect_T_hidden_and_logits(
    df_chunk: pd.DataFrame,
    tokenizer,
    model,
    lm_W,
    lm_b,  # may be None for LLaMA
    digit_ids: dict,
    device: str,
    batch_size: int = 8,
):
    """
    For P(T) prompts:
      - mean-pooled hidden states per layer (for PCA)
      - logit-lens over digits 1..4 at EACH layer
      - final-layer prediction over digits (P(T)_final)

    Returns:
      pooled_by_layer: dict[layer] -> np.array[N,D]
      layer_scores_T : dict[layer] with keys:
           pred_digit, correct_digit, logp_correct, logp_margin,
           entropy (predictive entropy), max_prob, probs_all (4-way)
      final_pred_T   : np.array[N]
      gold_T         : np.array[N]
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
        last_pos = attn_mask.sum(dim=1) - 1  # (B,)

        # ---- mean-pooled for PCA
        pooled = mean_pool_hidden(hidden_states, attn_mask)
        for l, vec in pooled.items():
            pooled_by_layer[l].append(vec)  # append (B,D)

        # ---- logit-lens for digits per layer
        num_layers = len(hidden_states)

        for layer_idx in range(num_layers):
            h = hidden_states[layer_idx]  # (B,T,D)

            # gather last position repr
            idx = last_pos.view(B, 1, 1).expand(-1, 1, h.size(-1))
            h_last = torch.gather(h, dim=1, index=idx).squeeze(1)  # (B,D)

            # logits from this layer: h_last @ W^T (+ bias if present)
            logits = torch.matmul(h_last, lm_W.T)
            if lm_b is not None:
                logits = logits + lm_b

            # keep only digit logits
            digit_logits = torch.stack([
                logits[:, digit_ids[1]],
                logits[:, digit_ids[2]],
                logits[:, digit_ids[3]],
                logits[:, digit_ids[4]],
            ], dim=-1)  # (B,4)

            log_probs = torch.log_softmax(digit_logits.float(), dim=-1)  # (B,4)
            probs = log_probs.exp()                                       # (B,4)

            # predicted digit (1..4) at this layer
            pred_idx = torch.argmax(log_probs, dim=-1)        # [0..3]
            pred_digit = (pred_idx + 1).cpu().numpy()         # [1..4]

            # correct index [0..3]
            gold_idx = (torch.tensor(batch_gold, device=log_probs.device) - 1).clamp(0, 3)
            logp_correct = log_probs[torch.arange(B, device=log_probs.device), gold_idx]

            # margin = logp(correct) - max_other
            mask = torch.ones_like(log_probs, dtype=torch.bool)
            mask[torch.arange(B), gold_idx] = False
            other_logp = log_probs.masked_fill(~mask, -1e9)
            max_other = other_logp.max(dim=-1).values
            logp_margin = logp_correct - max_other

            # predictive entropy and max prob
            entropy_batch = -(probs * log_probs).sum(dim=-1)  # (B,)
            max_prob, _ = probs.max(dim=-1)                   # (B,)

            # store
            layer_scores_T[layer_idx]["pred_digit"].extend(pred_digit.tolist())
            layer_scores_T[layer_idx]["correct_digit"].extend(batch_gold.tolist())
            layer_scores_T[layer_idx]["logp_correct"].extend(logp_correct.cpu().tolist())
            layer_scores_T[layer_idx]["logp_margin"].extend(logp_margin.cpu().tolist())
            layer_scores_T[layer_idx]["entropy"].extend(entropy_batch.cpu().tolist())
            layer_scores_T[layer_idx]["max_prob"].extend(max_prob.cpu().tolist())
            layer_scores_T[layer_idx]["probs_all"].extend(probs.cpu().numpy().tolist())

            if layer_idx == num_layers - 1:
                # final layer prediction used as "P(T)_final"
                final_pred_T_all.extend(pred_digit.tolist())

    # stack pooled
    for l in pooled_by_layer.keys():
        pooled_by_layer[l] = np.vstack(pooled_by_layer[l])  # (N,D)

    # finalize arrays
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
                        help="Output directory (subfolder {run_tag}_T will be created)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HF model name")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device, e.g., cuda:0")
    parser.add_argument("--max_samples", type=int, default=400,
                        help="Max rows to use (subsample if larger). Set big to use all.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for forward passes")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Torch dtype for model")
    parser.add_argument("--run_tag", type=str, default="run1",
                        help="Tag to separate multiple runs (run1/run2/run3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling + torch")
    parser.add_argument("--n_bins", type=int, default=15,
                        help="ECE bins")
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
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    if len(df) > args.max_samples:
        idx = rng.choice(df.index.values, size=args.max_samples, replace=False)
        df_T = df.loc[idx].reset_index(drop=True)
        print(f"[Module1] Subsampled to {len(df_T)} rows (max_samples={args.max_samples})")
    else:
        df_T = df.copy()
        print(f"[Module1] Using all {len(df_T)} rows (<= max_samples)")

    # ---------- Hugging Face login (optional) ----------
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("[Module1] Logging into Hugging Face Hub using HF_TOKEN / HUGGING_FACE_HUB_TOKEN")
        login(token=hf_token)
    else:
        print("[Module1] No HF token found in env; assuming prior `huggingface-cli login` or cached model.")

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map={"": args.device},   # single GPU keeps hidden-states local
    )
    model.eval()

    lm_head = model.lm_head
    lm_W = lm_head.weight
    lm_b = lm_head.bias
    if lm_b is None:
        print("[Module1] No bias term found in lm_head; setting lm_b = 0")
        lm_b = None  # handled in compute by skipping addition

    # Get token IDs for "1","2","3","4"
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

    # ---------- Per-layer metrics (overall + per-dialect) ----------
    print("[Module1] Computing per-layer proto accuracy + consistency + entropy/ECE")
    dialects = df_T["dialect"].tolist()
    dial_arr = np.array(dialects)

    # overall & per-dialect rows in one table (dialect field absent = overall)
    rows_T = []
    per_example_rows = []

    for l in sorted(layer_scores_T.keys()):
        sc = layer_scores_T[l]
        pred = sc["pred_digit"]
        gold = sc["correct_digit"]
        correct = (pred == gold).astype(np.int8)

        max_p = np.array(sc["max_prob"], dtype=np.float32)
        entropy = np.array(sc["entropy"], dtype=np.float32)

        same_as_final = (pred == final_pred_T).mean()
        proto_acc = correct.mean()
        ece_over = compute_ece(max_p, correct, n_bins=args.n_bins)

        rows_T.append({
            "layer": l,
            "ProtoAcc_T": proto_acc,
            "ConsistentWithFinal_T": same_as_final,
            "MeanLogP_margin_T": float(np.mean(sc["logp_margin"])),
            "MeanPredEntropy_T": float(entropy.mean()),
            "ECE_T": ece_over,
        })

        # per-dialect metrics
        for d in sorted(np.unique(dial_arr)):
            mask = (dial_arr == d)
            if mask.sum() == 0:
                continue
            rows_T.append({
                "layer": l,
                "dialect": d,
                "ProtoAcc_T_dialect": correct[mask].mean(),
                "MeanPredEntropy_T_dialect": float(entropy[mask].mean()),
                "ECE_T_dialect": compute_ece(max_p[mask], correct[mask], n_bins=args.n_bins),
                "ConsistentWithFinal_T_dialect": (pred[mask] == final_pred_T[mask]).mean(),
                "MeanLogP_margin_T_dialect": float(np.mean(sc["logp_margin"][mask])),
            })

        # per-example dump for this layer
        for i in range(len(pred)):
            per_example_rows.append({
                "run_tag": args.run_tag,
                "seed": args.seed,
                "layer": l,
                "row_id": int(i),                    # position within df_T
                "unique_key": str(df_T.loc[i, "unique_key"]),
                "dialect": str(df_T.loc[i, "dialect"]),
                "gold": int(gold[i]),
                "pred": int(pred[i]),
                "correct": int(correct[i]),
                "max_prob": float(max_p[i]),
                "entropy": float(entropy[i]),
                "probs": layer_scores_T[l]["probs_all"][i],  # 4-way probs
            })

    layer_metrics_T = pd.DataFrame(rows_T).sort_values(["layer", "dialect"], na_position="first")

    # ---------- PCA + geometry per layer ----------
    print("[Module1] Running PCA + geometry metrics per layer")
    N_T = len(df_T)
    pca_rows_T = []
    geom_rows_T = []

    for l in sorted(pooled_T.keys()):
        X = pooled_T[l]  # (N_T,D)

        # PCA 2D
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)

        # silhouette over dialects
        sil = silhouette_score(X2, dialects)

        # BCSS/WCSS
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
            "ExplainedVar_PC1_T": float(pca.explained_variance_ratio_[0]),
            "ExplainedVar_PC2_T": float(pca.explained_variance_ratio_[1]),
        })

        # PCA point rows (for plo
