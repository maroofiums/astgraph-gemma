import argparse
import numpy as np, pandas as pd
from .metrics import bootstrap_ci, paired_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/run_v3.jsonl")
ap.add_argument("--ref", default="repomap_sig")
ap.add_argument("--methods", default=None, help="comma list; restricts to these methods AND to instances present for all of them")
ap.add_argument("--plot", default=None)
a = ap.parse_args()
df = pd.read_json(a.results, lines=True)
for c in ("file_acc1", "file_acc3", "file_acc5", "func_acc1"):
    df[c] = df[c].astype(float)
df.loc[df.method == "bm25_files", "func_acc1"] = np.nan
if a.methods:
    ms = a.methods.split(",")
    df = df[df.method.isin(ms)]
    sets = [set(g.instance_id) for _, g in df.groupby("method")]
    common = set.intersection(*sets) if sets else set()
    df = df[df.instance_id.isin(common)]
    print(f"[restricted to {len(common)} instances present for all {len(ms)} methods]")

def ci(x):
    x = x.dropna().astype(float)
    m, lo, hi = bootstrap_ci(x)
    return f"{m:.3f} [{lo:.3f},{hi:.3f}]"

rows = []
for name, g in df.groupby("method"):
    hits = g.file_acc1.sum()
    rows.append(dict(method=name, n=len(g), file_acc1=ci(g.file_acc1), file_acc3=ci(g.file_acc3), func_acc1=ci(g.func_acc1),
                     ctx_tok=round(g.ctx_tokens.mean()), prompt_tok=round(g.prompt_tokens.mean()),
                     prefill_p50=round(g.prefill_ms.median()), e2e_p50_s=round(g.e2e_ms.median() / 1000, 2),
                     e2e_p95_s=round(g.e2e_ms.quantile(.95) / 1000, 2),
                     tok_per_hit=round(g.prompt_tokens.sum() / max(hits, 1))))
print(pd.DataFrame(rows).sort_values("method").to_string(index=False))

print(f"\nPaired comparisons vs {a.ref} (difference, 95% CI, raw p, Holm-adjusted p):")
ref = df[df.method == a.ref].set_index("instance_id")
res = []
for name, g in df.groupby("method"):
    if name == a.ref or name == "bm25_files" and False:
        continue
    g = g.set_index("instance_id"); common = g.index.intersection(ref.index)
    for col in ("file_acc1", "func_acc1"):
        x, y = g.loc[common, col], ref.loc[common, col]
        ok = x.notna() & y.notna()
        if ok.sum() < 5:
            continue
        d, lo, hi, p = paired_bootstrap(x[ok].astype(float), y[ok].astype(float))
        res.append([name, col, d, lo, hi, p, int(ok.sum())])
order = np.argsort([r[5] for r in res]); m = len(res); adj = {}; run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, res[i][5] * (m - rank))); adj[i] = run
for i, r in enumerate(res):
    print(f"  {r[0]:18s} {r[1]}: {r[2]:+.3f} [{r[3]:+.3f},{r[4]:+.3f}] p={r[5]:.3f} holm={adj[i]:.3f} (n={r[6]})")

if a.plot:
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        s = df.groupby("method").agg(tok=("ctx_tokens", "mean"), acc=("file_acc1", "mean")).reset_index()
        plt.figure(figsize=(6, 4)); plt.scatter(s.tok, s.acc)
        for _, r in s.iterrows(): plt.annotate(r.method, (r.tok, r.acc), fontsize=7)
        plt.xlabel("Mean context tokens"); plt.ylabel("File Acc@1"); plt.tight_layout(); plt.savefig(a.plot, dpi=200)
    except Exception as e:
        print("plot skipped:", e)
