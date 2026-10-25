"""GPU-free context-coverage diagnostic: is the gold file / function definition present in the prompt?"""
import re
from dataclasses import replace
from pathlib import Path
from astgraph.parser import extract_repo
from astgraph.graph import build_graph
from .config import registry
from .contexts import rank_nodes, file_bm25, graph_context, raw_context, SnippetIndex
from .data import gold_from_patch

RANKERS = ("bm25", "ppr", "bm25m", "hybrid")
BUDGETS = (512, 1024, 2048, 4096)

def has_def(body, leaf):
    return any(re.search(rf"(def\s+{re.escape(g)}\s*\(|class\s+{re.escape(g)}\b)", body) for g in leaf)

def coverage_for_instance(inst, repo_dir, count, trunc_issue, rankers=RANKERS, budgets=BUDGETS, pcache=None):
    syms = extract_repo(Path(repo_dir), cache=pcache); G = build_graph(syms)
    gf, gfun, _ = gold_from_patch(inst["patch"], syms)
    iss = trunc_issue(inst["problem_statement"])
    fr = file_bm25(repo_dir, iss, sorted({s.file for s in syms}))
    leaf = {g.split(".")[-1] for v in gfun.values() for g in v if g != "<module>"}
    snip = SnippetIndex(repo_dir, syms)
    sc = {rk: rank_nodes(G, iss, rk) for rk in rankers}
    row = dict(id=inst["instance_id"], repo=inst["repo"], has_func=bool(leaf),
               bm25_recall1=bool(set(fr[:1]) & gf), bm25_recall3=bool(set(fr[:3]) & gf), bm25_recall10=bool(set(fr[:10]) & gf))
    for B in budgets:
        reg = registry(B)
        for rk in rankers:
            for mname in ("repomap_sig", "ours"):
                mm = replace(reg[mname], ranker=rk)
                body, _, order = graph_context(G, sc[rk], mm, count)
                row[f"{mname}|{rk}|{B}|file"] = bool(set(order) & gf)
                row[f"{mname}|{rk}|{B}|func"] = has_def(body, leaf) if leaf else None
        body, _, files = snip.select(iss, count, B)
        row[f"snippet_rag||{B}|file"] = bool(set(files) & gf)
        row[f"snippet_rag||{B}|func"] = has_def(body, leaf) if leaf else None
        body, _ = raw_context(repo_dir, fr[:10], count, B)
        row[f"raw_topk||{B}|file"] = any(f in body for f in gf)
        row[f"raw_topk||{B}|func"] = has_def(body, leaf) if leaf else None
    return row
