
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 1 (T+EN): Layer-wise analysis using BOTH passages (P(TEN))

For each item:
  - target dialect passage: flores_passage  (T)
  - English passage:        flores_passage_en (looked up from eng_Latn row with same unique_key)
  - question + choices from the target row

Per layer we compute (old-style names, new entropy):
  * ProtoAcc_TEN
  * ConsistentWithFinal_TEN
  * MeanLogP_margin_TEN
  * Entropy_lang_TEN            (predictive entropy, avg over items)
  * Silhouette_lang_TEN
  * BCSS_WCSS_lang_TEN
  * ExplainedVar_PC1_TEN / PC2_TEN

Also:
  * Per-dialect proto accuracy        -> layer_metrics_TEN_by_dialect.csv
  * Per-dialect PCA centroids         -> pca_centroids_TEN.csv
  * Per-point PCA for plotting        -> pca_points_TEN.csv
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

EVAL_TEN = """Read the passage in target language ({dialect}) and in English, then answer the question.
Think step by step. Carefully evaluate all the choices and eliminate the incorrect options.
Select the single best-supported answer as your response.

Passage_T:
{flores_passage}

Passage_EN:
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

def build_prompt_TEN(row: pd.Series) -> str:
    return EVAL_TEN.format(**row.to_dict())

def build_chat_input(prompt: str) -> str:
    return f"<|system|>\n{SYSTEM_MSG}\n<|user|>\n{prompt}\n<|assistant|>\n"

@torch.no_grad()
def mean_pool_hidden(hidden_states, attn_mask):
    pooled = {}
    mask = attn_mask.unsqueeze(-1)
    for l, h in enumerate(hidden_states):
        denom = mask.sum(dim=1).clamp(min=1)
        pooled[l] = ((h * mask).sum(dim=1) / denom).detach().cpu().numpy()
    return pooled

@torch.no_grad()
def collect_TEN_hidden_and_logits(
    df_chunk: pd.DataFrame,
    tokenizer,
    model,
    lm_W,
    lm_b,
    digit_ids,
    device: str,
    batch_size: int = 8,
):
    prompts = [build_chat_input(build_prompt_TEN(r)) for _, r in df_chunk.iterrows()]
    gold = df_chunk["correct_answer_num"].astype(int).values

    pooled_by_layer = defaultdict(list)
    layer_scores = defaultdict(lambda: defaultdict(list))
    final_pred_all = []

    print(f"[Module1-TEN] Collecting hidden states + logits for {len(prompts)} examples (P(TEN))")

    for start in tqdm(range(0, len(prompts), batch_size), desc="[Module1-TEN] Forward batches"):
        bp = prompts[start:start+batch_size]
        bg = gold[start:start+batch_size]

        enc = tokenizer(bp, return_tensors="pt", padding=True, truncation=True).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
        attn_mask = enc["attention_mask"]

        B, T = attn_mask.shape
        last_pos = attn_mask.sum(dim=1) - 1

        # pooled for PCA
        pooled = mean_pool_hidden(hidden_states, attn_mask)
        for l, v in pooled.items():
            pooled_by_layer[l].append(v)

        # logit-lens each layer
        num_layers = len(hidden_states)
        for l in range(num_layers):
            h = hidden_states[l]
            idx = last_pos.view(B,1,1).expand(-1,1,h.size(-1))
            h_last = torch.gather(h, dim=1, index=idx).squeeze(1)

            logits = torch.matmul(h_last, lm_W.T)
            if isinstance(lm_b, torch.Tensor):
                logits = logits + lm_b

            digit_logits = torch.stack(
                [logits[:, digit_ids[1]], logits[:, digit_ids[2]], logits[:, digit_ids[3]], logits[:, digit_ids[4]]],
                dim=-1
            )
            log_probs = torch.log_softmax(digit_logits.float(), dim=-1)
            probs = log_probs.exp()
            entropy_batch = -(probs * log_probs).sum(dim=-1)  # predictive entropy

            pred_idx = torch.argmax(log_probs, dim=-1)
            pred_digit = (pred_idx + 1).cpu().numpy()

            gold_idx = (torch.tensor(bg, device=log_probs.device) - 1).clamp(0,3)
            logp_correct = log_probs[torch.arange(B, device=log_probs.device), gold_idx]

            mask = torch.ones_like(log_probs, dtype=torch.bool)
            mask[torch.arange(B), gold_idx] = False
            other_logp = log_probs.masked_fill(~mask, -1e9)
            max_other = other_logp.max(dim=-1).values
            logp_margin = logp_correct - max_other

            layer_scores[l]["pred_digit"].extend(pred_digit.tolist())
            layer_scores[l]["correct_digit"].extend(bg)
            layer_scores[l]["logp_correct"].extend(logp_correct.cpu().tolist())
            layer_scores[l]["logp_margin"].extend(logp_margin.cpu().tolist())
            layer_scores[l]["entropy"].extend(entropy_batch.cpu().tolist())

            if l == num_layers - 1:
                final_pred_all.extend(pred_digit.tolist())

    for l in pooled_by_layer.keys():
        pooled_by_layer[l] = np.vstack(pooled_by_layer[l])
    for l, dd in layer_scores.items():
        for k in dd.keys():
            layer_scores[l][k] = np.array(dd[k])

    final_pred = np.array(final_pred_all)
    return pooled_by_layer, layer_scores, final_pred, gold

def main():
    parser = argparse.ArgumentParser(description="Module 1 (TEN): Layer-wise P(TEN) analysis")
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--out_dir",  type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--device",    type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16", choices=["bfloat16","float16","float32"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Module1-TEN] Loading data from: {args.data_csv}")
    df = pd.read_csv(args.data_csv)

    needed = ["unique_key","dialect","flores_passage","question",
              "mc_answer1","mc_answer2","mc_answer3","mc_answer4","correct_answer_num"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df = df[df["correct_answer_num"].isin([1,2,3,4])].reset_index(drop=True)
    print(f"[Module1-TEN] Total rows with valid correct_answer_num ∈ {{1..4}}: {len(df)}")

    # Build English passage map
    eng_map = dict(zip(
        df.loc[df["dialect"]=="eng_Latn","unique_key"].astype(str),
        df.loc[df["dialect"]=="eng_Latn","flores_passage"]
    ))
    df["flores_passage_en"] = df["unique_key"].astype(str).map(eng_map)
    df = df[~df["flores_passage_en"].isna()].reset_index(drop=True)
    print(f"[Module1-TEN] Rows with English passage available: {len(df)}")

    rng = np.random.RandomState(42)
    if len(df) > args.max_samples:
        idx = rng.choice(df.index.values, size=args.max_samples, replace=False)
        df_TEN = df.loc[idx].reset_index(drop=True)
        print(f"[Module1-TEN] Subsampled to {len(df_TEN)} rows (max_samples={args.max_samples})")
    else:
        df_TEN = df.copy()
        print(f"[Module1-TEN] Using all {len(df_TEN)} rows (<= max_samples)")

    # HF Hub (optional)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("[Module1-TEN] Logging into Hugging Face Hub using HF_TOKEN / HUGGING_FACE_HUB_TOKEN")
        login(token=hf_token)
    else:
        print("[Module1-TEN] No HF token found; assuming cached login/models.")

    print(f"[Module1-TEN] Loading model: {args.model_name}")
    print(f"[Module1-TEN] Device: {args.device}, dtype: {args.dtype}")

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype, device_map={"": args.device}
    )
    model.eval()

    lm_W = model.lm_head.weight
    lm_b = model.lm_head.bias
    if lm_b is None:
        print("[Module1-TEN] No bias term found in lm_head; setting lm_b = 0")
        lm_b = torch.tensor(0.0, device=lm_W.device, dtype=lm_W.dtype)

    print("[Module1-TEN] Resolving digit token IDs for labels 1,2,3,4")
    digit_ids = {}
    with torch.no_grad():
        for k in [1,2,3,4]:
            ids = tokenizer(str(k), add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                print(f"[Module1-TEN][WARN] Label {k} not single token: {ids}")
            digit_ids[k] = ids[0]
    print(f"[Module1-TEN] Digit IDs: {digit_ids}")

    # Collect hidden + logits
    pooled_TEN, layer_scores_TEN, final_pred_TEN, gold_TEN = collect_TEN_hidden_and_logits(
        df_TEN, tokenizer, model, lm_W, lm_b, digit_ids, args.device, batch_size=args.batch_size
    )
    num_layers = len(layer_scores_TEN)
    print(f"[Module1-TEN] Collected hidden/logits for {num_layers} layers (including embedding layer 0)")

    # Per-layer metrics
    print("[Module1-TEN] Computing per-layer proto accuracy + consistency vs final layer")
    rows = []
    for l in sorted(layer_scores_TEN.keys()):
        sc = layer_scores_TEN[l]
        pred = sc["pred_digit"]; gold = sc["correct_digit"]
        rows.append({
            "layer": l,
            "ProtoAcc_TEN": (pred == gold).mean(),
            "ConsistentWithFinal_TEN": (pred == final_pred_TEN).mean(),
            "MeanLogP_margin_TEN": float(np.mean(sc["logp_margin"])),
            "Entropy_lang_TEN": float(np.mean(sc["entropy"])),
        })
    layer_metrics_TEN = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)

    # Per-dialect accuracy
    print("[Module1-TEN] Computing per-layer, per-dialect proto accuracy (P(TEN))")
    dialects = df_TEN["dialect"].tolist()
    dial_arr = np.array(dialects)
    rows_d = []
    for l in sorted(layer_scores_TEN.keys()):
        sc = layer_scores_TEN[l]
        correct = (sc["pred_digit"] == sc["correct_digit"]).astype(np.int8)
        for d in sorted(np.unique(dial_arr)):
            m = (dial_arr == d)
            if m.sum() == 0: continue
            rows_d.append({"layer": l, "dialect": d, "ProtoAcc_TEN_dialect": float(correct[m].mean())})
    layer_metrics_TEN_by_dialect = pd.DataFrame(rows_d).sort_values(["layer","dialect"])

    # PCA + geometry
    print("[Module1-TEN] Running PCA + geometry metrics per layer")
    N = len(df_TEN)
    pca_rows, cent_rows, geom_rows = [], [], []
    for l in sorted(pooled_TEN.keys()):
        X = pooled_TEN[l]
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)

        sil = silhouette_score(X2, dialects)
        # BCSS/WCSS
        overall = X2.mean(axis=0)
        langs = np.unique(dialects)
        bcss = wcss = 0.0
        for d in langs:
            idxs = [i for i, la in enumerate(dialects) if la == d]
            pts = X2[idxs]; ctr = pts.mean(axis=0)
            bcss += len(pts) * np.linalg.norm(ctr - overall)**2
            wcss += ((pts - ctr)**2).sum()
        bcss_wcss = bcss / max(wcss, 1e-6)

        geom_rows.append({
            "layer": l,
            "Silhouette_lang_TEN": sil,
            "BCSS_WCSS_lang_TEN": bcss_wcss,
            "ExplainedVar_PC1_TEN": float(pca.explained_variance_ratio_[0]),
            "ExplainedVar_PC2_TEN": float(pca.explained_variance_ratio_[1]),
        })

        for i in range(N):
            pca_rows.append({
                "layer": l, "pc1": X2[i,0], "pc2": X2[i,1],
                "dialect": dialects[i],
                "correct_final": int(gold_TEN[i] == final_pred_TEN[i]),
            })

        for d in langs:
            idxs = np.where(np.array(dialects)==d)[0]
            if len(idxs)==0: continue
            pts = X2[idxs]; ctr = pts.mean(axis=0)
            within_var = float(((pts-ctr)**2).sum()/max(len(pts)-1,1))
            cent_rows.append({
                "layer": l, "dialect": d,
                "pc1_mean": float(ctr[0]), "pc2_mean": float(ctr[1]),
                "count": int(len(idxs)), "within_var": within_var
            })

        print(f"[Module1-TEN] Layer {l:02d}: Sil={sil:.3f}, Entropy={float(np.mean(layer_scores_TEN[l]['entropy'])):.3f}, BCSS/WCSS={bcss_wcss:.3f}")

    geom = pd.DataFrame(geom_rows).sort_values("layer").reset_index(drop=True)
    pca_pts = pd.DataFrame(pca_rows)
    pca_cent = pd.DataFrame(cent_rows)
    layer_metrics_TEN_full = layer_metrics_TEN.merge(geom, on="layer", how="left").sort_values("layer")

    # Save
    out_metrics      = os.path.join(args.out_dir, "layer_metrics_TEN.csv")
    out_metrics_d    = os.path.join(args.out_dir, "layer_metrics_TEN_by_dialect.csv")
    out_pca_pts      = os.path.join(args.out_dir, "pca_points_TEN.csv")
    out_pca_cent     = os.path.join(args.out_dir, "pca_centroids_TEN.csv")

    layer_metrics_TEN_full.to_csv(out_metrics, index=False)
    layer_metrics_TEN_by_dialect.to_csv(out_metrics_d, index=False)
    pca_pts.to_csv(out_pca_pts, index=False)
    pca_cent.to_csv(out_pca_cent, index=False)

    print(f"[Module1-TEN] Saved per-dialect metrics to: {out_metrics_d}")
    print(f"[Module1-TEN] Saved layer-wise metrics to: {out_metrics}")
    print(f"[Module1-TEN] Saved PCA centroids to     : {out_pca_cent}")
    print(f"[Module1-TEN] Saved PCA points to        : {out_pca_pts}")
    print("[Module1-TEN] Done.")

if __name__ == "__main__":
    main()
