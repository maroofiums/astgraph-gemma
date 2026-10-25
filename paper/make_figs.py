"""Regenerates the paper figures from results/*.csv and results/run_v3.jsonl."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench.metrics import bootstrap_ci

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "violet": "#6250d6", "ink": "#0b0b0b", "mute": "#52514e", "grid": "#e4e3df"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["mute"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["mute"], "ytick.color": C["mute"], "axes.grid": True,
                     "grid.color": C["grid"], "grid.linewidth": .6, "axes.axisbelow": True, "legend.frameon": False})
OUT = ["paper/figs", "docs/img"]
def save(fig, name):
    for d in OUT:
        fig.savefig(f"{d}/{name}.pdf" if d == "paper/figs" else f"{d}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

df = pd.read_json("results/run_v3.jsonl", lines=True)
for c in ("file_acc1", "func_acc1"): df[c] = df[c].astype(float)

# ---- Fig 1: main comparison with CIs
meth = [("issue_only", "issue only"), ("bm25_files", "BM25\nfiles\n(no LLM)"), ("raw_topk", "raw\ntop-k"), ("snippet_rag", "snippet\nRAG"), ("ours", "ours\n+edges\n+flow"), ("repomap_sig", "signature\nmap")]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for a, col, ttl in zip(ax, ("file_acc1", "func_acc1"), ("File Acc@1", "Function Acc@1")):
    for i, (m, lab) in enumerate(meth):
        x = df[df.method == m][col].dropna()
        if m == "bm25_files" and col == "func_acc1": continue
        mu, lo, hi = bootstrap_ci(x)
        a.bar(i, mu, color=C["blue"] if m in ("ours", "repomap_sig") else "#9aa0a6", width=.62)
        a.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color=C["ink"], lw=1, capsize=2.5)
        a.text(i, hi + .012, f"{mu:.2f}", ha="center", fontsize=7.5, color=C["ink"])
    a.set_xticks(range(len(meth))); a.set_xticklabels([l for _, l in meth], fontsize=6.5)
    a.set_ylabel(ttl); a.set_ylim(0, .72 if col == "file_acc1" else .4); a.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "fig_main")

# ---- Fig 2: dev coverage vs budget
cov = pd.read_csv("results/coverage_dev.csv")
B = [512, 1024, 2048, 4096]
mean = lambda c: cov[c].dropna().astype(float).mean()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
sty = {"bm25": (C["blue"], "o", "-"), "ppr": (C["orange"], "s", "-"), "bm25m": (C["aqua"], "^", "-"), "hybrid": (C["violet"], "D", "-")}
for a, kind, ttl in zip(ax, ("file", "func"), ("gold file in prompt", "gold function in prompt")):
    for rk, (c, mk, ls) in sty.items():
        a.plot(B, [mean(f"repomap_sig|{rk}|{b}|{kind}") for b in B], color=c, marker=mk, ls=ls, ms=4.5, lw=1.6, label=f"signatures, {rk}")
    for nm, c, mk in (("snippet_rag", C["mute"], "x"), ("raw_topk", "#9aa0a6", "+")):
        a.plot(B, [mean(f"{nm}||{b}|{kind}") for b in B], color=c, marker=mk, ls="--", ms=5, lw=1.3, label=nm.replace("_", " "))
    a.set_xscale("log", base=2); a.set_xticks(B); a.set_xticklabels(B); a.set_xlabel("context budget (tokens)"); a.set_ylabel(ttl)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(.5, -.16))
fig.tight_layout(); save(fig, "fig_coverage")

# ---- Fig 3: budget sweep
sw = pd.read_csv("results/sweep.csv")
io = df[df.method == "issue_only"].iloc[:0]
first60 = df[df.method == "ours"].instance_id.unique()  # order of insertion = TEST order
ids = list(dict.fromkeys(df[df.method == "ours"].instance_id))[:60]
iom = df[(df.method == "issue_only") & df.instance_id.isin(ids)]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
st = {"repomap_sig": (C["blue"], "o", "signature map"), "ours": (C["orange"], "s", "ours (+edges, flow)"), "snippet_rag": (C["mute"], "x", "snippet RAG")}
for a, col, ttl in zip(ax, ("file_acc1", "func_acc1"), ("File Acc@1", "Function Acc@1")):
    for m, (c, mk, lab) in st.items():
        g = sw[sw.method == m]; a.plot(g.prompt_tok, g[col], color=c, marker=mk, ms=5, lw=1.6, label=lab)
    a.axhline(iom[col].mean(), color="#9aa0a6", ls=":", lw=1.4, label="issue only")
    a.set_xlabel("mean prompt tokens"); a.set_ylabel(ttl)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.8, bbox_to_anchor=(.5, -.1))
fig.tight_layout(); save(fig, "fig_sweep")

# ---- Fig 4: memory
mem = pd.read_csv("results/memory.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
a = ax[0]
a.plot(mem["T"], mem.kv_act_mb, color=C["blue"], marker="o", ms=4.5, lw=1.6, label="measured (KV + activations)")
a.plot(mem["T"], mem.analytic_kv_mib, color=C["orange"], marker="s", ms=4.5, lw=1.6, label="analytic KV, fp16")
a.set_yscale("log"); a.set_xlabel("prompt tokens"); a.set_ylabel("MiB above weights"); a.legend(fontsize=6.8, loc="lower right")
probe = {2048: (2172, 728, 347), 4096: (4340, 2069, 624), 6144: (None, 4026, 909), 8192: (None, None, 1191)}
a = ax[1]; w = .26; modes = [("full logits", C["orange"]), ("last-token logits", C["blue"]), ("chunked (512) + last logit", C["aqua"])]
for j, (lab, c) in enumerate(modes):
    for i, T in enumerate(probe):
        v = probe[T][j]
        if v is None:
            a.text(i + (j - 1) * w, 40, "OOM", rotation=90, ha="center", va="bottom", fontsize=6.3, color=C["mute"])
        else:
            a.bar(i + (j - 1) * w, v, width=w * .92, color=c, label=lab if i == 0 else None)
a.set_xticks(range(4)); a.set_xticklabels(list(probe)); a.set_xlabel("prompt tokens"); a.set_ylabel("MiB above weights"); a.set_ylim(0, 6600); a.legend(fontsize=6.3, loc="upper left"); a.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "fig_memory")
print("ok")
