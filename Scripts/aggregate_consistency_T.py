#!/usr/bin/env python
import os, glob, json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Point to the directory that contains run1_T/, run2_T/, run3_T/
BASE = "/scratch/ekata/projects/mcqa/mcqa-synergy/results"

per_example_files = sorted(glob.glob(os.path.join(BASE, "run*_T/per_example_T.csv")))
assert per_example_files, "No per_example_T.csv files found."

dfs = []
for f in per_example_files:
    df = pd.read_csv(f)
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)

# We expect rows grouped by (layer, row_id) or (layer, unique_key, dialect)
group_cols = ["layer", "row_id", "dialect", "unique_key"]

def consistency_rate(preds):
    # fraction of runs that agree on the mode class
    if len(preds) == 0: return np.nan
    c = Counter(preds)
    return max(c.values()) / len(preds)

rows = []
for (layer, dialect), g_lang in all_df.groupby(["layer", "dialect"]):
    # overall per dialect: consistency of predictions
    cons = g_lang.groupby("row_id")["pred"].apply(lambda s: consistency_rate(list(s))).mean()
    # accuracy & confidence averages per dialect per layer
    acc  = g_lang.groupby("row_id")["correct"].mean().mean()
    conf = g_lang.groupby("row_id")["max_prob"].mean().mean()
    ent  = g_lang.groupby("row_id")["entropy"].mean().mean()

    rows.append({
        "layer": int(layer),
        "dialect": dialect,
        "ConsistencyAcrossRuns_T": float(cons),  # 0..1
        "AccAcrossRuns_T": float(acc),           # mean of run-wise correctness
        "MeanConfAcrossRuns_T": float(conf),
        "MeanEntropyAcrossRuns_T": float(ent),
    })

cons_dialect = pd.DataFrame(rows).sort_values(["layer","dialect"])

# overall (collapse dialect)
rows_all = []
for layer, g in all_df.groupby("layer"):
    cons = g.groupby(["row_id"])["pred"].apply(lambda s: consistency_rate(list(s))).mean()
    acc  = g.groupby(["row_id"])["correct"].mean().mean()
    conf = g.groupby(["row_id"])["max_prob"].mean().mean()
    ent  = g.groupby(["row_id"])["entropy"].mean().mean()

    rows_all.append({
        "layer": int(layer),
        "ConsistencyAcrossRuns_T": float(cons),
        "AccAcrossRuns_T": float(acc),
        "MeanConfAcrossRuns_T": float(conf),
        "MeanEntropyAcrossRuns_T": float(ent),
    })

cons_overall = pd.DataFrame(rows_all).sort_values("layer")

# save
cons_overall.to_csv(os.path.join(BASE, "consistency_overall_T.csv"), index=False)
cons_dialect.to_csv(os.path.join(BASE, "consistency_by_dialect_T.csv"), index=False)

print("Wrote:")
print(" -", os.path.join(BASE, "consistency_overall_T.csv"))
print(" -", os.path.join(BASE, "consistency_by_dialect_T.csv"))
