#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 1 (EN): Layer-wise analysis using the ENGLISH passage (P(EN))

- For each item, we take the English passage from the eng_Latn row that shares the same unique_key.
- We keep the target-dialect question and choices (same behavior as your earlier EN prompt).
- Builds P(EN) prompts and runs logit-lens over all layers.

Per layer we compute:
    * ProtoAcc_EN
    * ConsistentWithFinal_EN
    * MeanLogP_margin_EN
    * Silhouette_lang_EN (PCA over dialects)
    * BCSS_WCSS_lang_EN
    * Entropy_lang_EN  (predictive entropy, Beyond-the-Final-Layer style)
    * ExplainedVar_PC1_EN / PC2_EN
Plus:
    * Per-dialect proto accuracy (layer_metrics_EN_by_dialect.csv)
    * Per-dialect PCA centroids (pca_centroids_EN.csv)

Outputs (in --out_dir):
    layer_metrics_EN.csv
    layer_metrics_EN_by_dialect.csv
    pca_points_EN.csv
    pca_centroids_EN.csv
"""

import os
import argparse
from collections import defaultdict

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

EVAL_EN = """Read the passage thoroughly and answer the question based on the information provided.
Lets think step by step.
Carefully evaluate all the choices and eliminate the incorrect options.
Select the single best-supported answer as your response.

Passage:
{flores_passage_en}

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


def build_prompt_EN(row: pd.Series) -> str:
    return EVAL_EN.format(**row.to_dict())


def build_chat_input(prompt: str) -> str:
    # Keep your old chat-format
    return f"<|system|>\n{SYSTEM_MSG}\n<|user|>\n{prompt}\n<|assistant|>\n"


# -----------------------
# Helpers
# -----------------------

@torch.no_grad()
def mean_pool_hidden(hidden_states, attn_mask):
    pooled = {}
    mask = attn_mask.unsqueeze(-1)  # (B,T,1)
    for layer_idx, h in enumerate(hidden_states):
        denom = mask.sum(dim=1).clamp(min=1)
        m = (h * mask).sum(dim=1) / denom
        pooled[layer_idx] = m.detach().cpu().numpy()
    return pooled


@torch.no_grad()
def collect_EN_hidden_and_logits(
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
    For P(EN) prompts:
      - mean-pooled hidden states per layer (for PCA)
      - logit-lens over digits 1..4 at EACH layer
      - final-layer prediction over digits (P(EN)_final)
      - predictive entropy per example per layer (Beyond-the-Final-Layer)
    """
    prompts = [build_chat_input(build_prompt_EN(r)) for _, r in df_chunk.iterrows()]
    gold = df_chunk["correct_answer_num"].astype(int).values

    pooled_by_layer = defaultdict(list)
    layer_scores = defaultdict(lambda: defaultdict(list))
    final_pred_all = []

    print(f"[Module1-EN] Collecting hidden states + logits for {len(prompts)} examples (P(EN))")

    for start in tqdm(range(0, len(prompts), batch_size), desc="[Module1-EN] Forward batches"):
        batch_prompts = prompts[start:start + batch_size]
        batch_gold = gold[start:start + batch_size]

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
        hidden_states = out.hidden_states
        attn_mask = enc["attention_mask"]

        B, T = attn_mask.shape
        last_pos = attn_mask.sum(dim=1) - 1

        # mean-pooled for PCA
        pooled = mean_pool_hidden(hidden_states, attn_mask)
        for l, vec in pooled.items():
            pooled_by_layer[l].append(vec)

        # logit-lens for digits per layer
        num_layers = len(hidden_states)

        for layer_idx in range(num_layers):
            h = hidden_states[layer_idx]
            idx = last_pos.view(B, 1, 1).expand(-1, 1, h.size(-1))
            h_last = torch.gather(h, dim=1, index=idx).squeeze(1)  # (B,D)

            logits = torch.matmul(h_last, lm_W.T)
            if isinstance(lm_b, torch.Tensor):
                logits = logits + lm_b

            digit_logits = torch.stack([
                logits[:, digit_ids[1]],
                logits[:, digit_ids[2]],
                logits[:, digit_ids[3]],
                logits[:, digit_ids[4]],
            ], dim=-1)  # (B,4)

            log_probs = torch.log_softmax(digit_logits.float(), dim=-1)  # (B,4)
            probs = log_probs.exp()
            entropy_batch = -(probs * log_probs).sum(dim=-1)             # predictive entropy (B,)

            pred_idx = torch.argmax(log_probs, dim=-1)   # 0..3
            pred_digit = (pred_idx + 1).cpu().numpy()    # 1..4

            gold_idx = (torch.tensor(batch_gold, device=log_probs.device) - 1).clamp(0, 3)
            logp_correct = log_probs[torch.arange(B, device=log_probs.device), gold_idx]

            mask = torch.ones_like(log_probs, dtype=torch.bool)
            mask[torch.arange(B), gold_idx] = False
            other_logp = log_probs.masked_fill(~mask, -1e9)
            max_other = other_logp.max(dim=-1).values
            logp_margin = logp_correct - max_other

            # store
            layer_scores[layer_idx]["pred_digit"].extend(pred_digit.tolist())
            layer_scores[layer_idx]["correct_digit"].extend(batch_gold.tolist())
            layer_scores[layer_idx]["logp_correct"].extend(logp_correct.cpu().tolist())
            layer_scores[layer_idx]["logp_margin"].extend(logp_margin.cpu().tolist())
            layer_scores[layer_idx]["entropy"].extend(entropy_batch.cpu().tolist())

            if layer_idx == num_layers - 1:
                final_pred_all.extend(pred_digit.tolist())

    for l in pooled_by_layer.keys():
        pooled_by_layer[l] = np.vstack(pooled_by_layer[l])

    for l, dd in layer_scores.items():
        for k in dd.keys():
            layer_scores[l][k] = np.array(dd[k])

    final_pred = np.array(final_pred_all)
    return pooled_by_layer, layer_scores, final_pred, gold


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Module 1 (EN): Layer-wise P(EN) analysis")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Belebele-style CSV (needs: unique_key, dialect, flores_passage, question, mc_answer1..4, correct_answer_num)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load data ----------
    print(f"[Module1-EN] Loading data from: {args.data_csv}")
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
    print(f"[Module1-EN] Total rows with valid correct_answer_num ∈ {{1..4}}: {len(df)}")

    # --- build English passage lookup from eng_Latn rows ---
    eng_passage_df = df[df["dialect"] == "eng_Latn"][["unique_key", "flores_passage"]].copy()
    eng_map = dict(zip(eng_passage_df["unique_key"].astype(str), eng_passage_df["flores_passage"]))
    df["unique_key_str"] = df["unique_key"].astype(str)
    df["flores_passage_en"] = df["unique_key_str"].map(eng_map)

    # drop rows with no English passage available
    df = df[~df["flores_passage_en"].isna()].reset_index(drop=True)
    print(f"[Module1-EN] Rows with English passage available: {len(df)}")

    # Subsample
    rng = np.random.RandomState(42)
    if len(df) > args.max_samples:
        idx = rng.choice(df.index.values, size=args.max_samples, replace=False)
        df_EN = df.loc[idx].reset_index(drop=True)
        print(f"[Module1-EN] Subsampled to {len(df_EN)} rows (max_samples={args.max_samples})")
    else:
        df_EN = df.copy()
        print(f"[Module1-EN] Using all {len(df_EN)} rows (<= max_samples)")

    # ---------- HF login (optional) ----------
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("[Module1-EN] Logging into Hugging Face Hub using HF_TOKEN / HUGGING_FACE_HUB_TOKEN")
        login(token=hf_token)
    else:
        print("[Module1-EN] No HF token found; assuming cached login/models.")

    # ---------- Load model ----------
    print(f"[Module1-EN] Loading model: {args.model_name}")
    print(f"[Module1-EN] Device: {args.device}, dtype: {args.dtype}")

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
        device_map={"": args.device},
    )
    model.eval()

    lm_head = model.lm_head
    lm_W = lm_head.weight
    lm_b = lm_head.bias
    if lm_b is None:
        print("[Module1-EN] No bias term found in lm_head; setting lm_b = 0")
        lm_b = torch.tensor(0.0, device=lm_W.device, dtype=lm_W.dtype)

    # token IDs for "1".."4"
    print("[Module1-EN] Resolving digit token IDs for labels 1,2,3,4")
    digit_ids = {}
    with torch.no_grad():
        for k in [1, 2, 3, 4]:
            ids = tokenizer(str(k), add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                print(f"[Module1-EN][WARN] Label {k} is not a single token: {ids}")
            digit_ids[k] = ids[0]
    print(f"[Module1-EN] Digit IDs: {digit_ids}")

    # ---------- Collect hidden states + logits ----------
    pooled_EN, layer_scores_EN, final_pred_EN, gold_EN = collect_EN_hidden_and_logits(
        df_EN, tokenizer, model, lm_W, lm_b, digit_ids, args.device, batch_size=args.batch_size
    )

    num_layers = len(layer_scores_EN)
    print(f"[Module1-EN] Collected hidden/logits for {num_layers} layers (including embedding layer 0)")

    # ---------- Per-layer proto accuracy + consistency ----------
    print("[Module1-EN] Computing per-layer proto accuracy + consistency vs final layer")
    rows_EN = []
    for l in sorted(layer_scores_EN.keys()):
        sc = layer_scores_EN[l]
        pred = sc["pred_digit"]
        gold = sc["correct_digit"]
        proto_acc = (pred == gold).mean()
        same_as_final = (pred == final_pred_EN).mean()
        mean_margin = float(np.mean(sc["logp_margin"]))
        mean_entropy = float(np.mean(sc["entropy"]))  # predictive entropy

        rows_EN.append({
            "layer": l,
            "ProtoAcc_EN": proto_acc,
            "ConsistentWithFinal_EN": same_as_final,
            "MeanLogP_margin_EN": mean_margin,
            "Entropy_lang_EN": mean_entropy,  # name kept parallel to T
        })

    layer_metrics_EN = pd.DataFrame(rows_EN).sort_values("layer").reset_index(drop=True)

    # ---------- NEW: per-dialect proto accuracy ----------
    print("[Module1-EN] Computing per-layer, per-dialect proto accuracy (P(EN))")
    dialects = df_EN["dialect"].tolist()
    dial_arr = np.array(dialects)
    rows_EN_dialect = []
    for l in sorted(layer_scores_EN.keys()):
        sc = layer_scores_EN[l]
        pred = sc["pred_digit"]
        gold = sc["correct_digit"]
        correct = (pred == gold).astype(np.int8)
        for d in sorted(np.unique(dial_arr)):
            mask = (dial_arr == d)
            if mask.sum() == 0:
                continue
            rows_EN_dialect.append({
                "layer": l,
                "dialect": d,
                "ProtoAcc_EN_dialect": float(correct[mask].mean()),
            })
    layer_metrics_EN_by_dialect = pd.DataFrame(rows_EN_dialect).sort_values(["layer","dialect"])

    # ---------- PCA + geometry per layer ----------
    print("[Module1-EN] Running PCA + geometry metrics per layer")
    N = len(df_EN)
    pca_rows_EN = []
    pca_centroids_rows = []
    geom_rows_EN = []

    for l in sorted(pooled_EN.keys()):
        X = pooled_EN[l]  # (N,D)

        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)

        sil = silhouette_score(X2, dialects)

        # predictive entropy already per-example; average for this layer
        entropy = float(np.mean(layer_scores_EN[l]["entropy"]))

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

        geom_rows_EN.append({
            "layer": l,
            "Silhouette_lang_EN": sil,
            "BCSS_WCSS_lang_EN": bcss_wcss,
            "ExplainedVar_PC1_EN": float(pca.explained_variance_ratio_[0]),
            "ExplainedVar_PC2_EN": float(pca.explained_variance_ratio_[1]),
        })

        # per-point for plotting
        for i in range(N):
            pca_rows_EN.append({
                "layer": l,
                "pc1": X2[i, 0],
                "pc2": X2[i, 1],
                "dialect": dialects[i],
                "correct_final": int(gold_EN[i] == final_pred_EN[i]),
            })

        # per-dialect centroids
        for d in np.unique(dialects):
            idxs = np.where(np.array(dialects) == d)[0]
            if len(idxs) == 0:
                continue
            pts = X2[idxs]
            center = pts.mean(axis=0)
            within_var = float(((pts - center) ** 2).sum() / max(len(pts) - 1, 1))
            pca_centroids_rows.append({
                "layer": l,
                "dialect": d,
                "pc1_mean": float(center[0]),
                "pc2_mean": float(center[1]),
                "count": int(len(idxs)),
                "within_var": within_var,
            })

        print(f"[Module1-EN] Layer {l:02d}: Sil={sil:.3f}, Entropy={entropy:.3f}, BCSS/WCSS={bcss_wcss:.3f}")

    geom_EN = pd.DataFrame(geom_rows_EN).sort_values("layer").reset_index(drop=True)
    pca_EN = pd.DataFrame(pca_rows_EN)
    pca_centroids_EN = pd.DataFrame(pca_centroids_rows)

    layer_metrics_EN_full = layer_metrics_EN.merge(geom_EN, on="layer", how="left").sort_values("layer")

    # ---------- Save ----------
    out_metrics = os.path.join(args.out_dir, "layer_metrics_EN.csv")
    out_metrics_d = os.path.join(args.out_dir, "layer_metrics_EN_by_dialect.csv")
    out_pca_pts = os.path.join(args.out_dir, "pca_points_EN.csv")
    out_pca_cent = os.path.join(args.out_dir, "pca_centroids_EN.csv")

    layer_metrics_EN_full.to_csv(out_metrics, index=False)
    layer_metrics_EN_by_dialect.to_csv(out_metrics_d, index=False)
    pca_EN.to_csv(out_pca_pts, index=False)
    pca_centroids_EN.to_csv(out_pca_cent, index=False)

    print(f"[Module1-EN] Saved per-dialect metrics to: {out_metrics_d}")
    print(f"[Module1-EN] Saved layer-wise metrics to: {out_metrics}")
    print(f"[Module1-EN] Saved PCA centroids to     : {out_pca_cent}")
    print(f"[Module1-EN] Saved PCA points to        : {out_pca_pts}")
    print("[Module1-EN] Done.")


if __name__ == "__main__":
    main()
