from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Method:
    name: str
    kind: str                 # issue_only | bm25 | raw_topk | snippet_rag | graph
    fmt: str = "compact"      # compact | json
    edges: bool = True
    control: bool = True
    ranker: str = "hybrid"    # bm25 | ppr | bm25m | hybrid
    two_stage: bool = False
    budget: int = 2048
    file_order: str = "relevance"   # relevance | alpha | shuffle

MAIN = ["issue_only", "bm25_files", "snippet_rag", "raw_topk", "repomap_sig", "ours"]
ABLATE = ["sig_edges", "sig_control", "abl_json", "abl_alpha_order", "abl_shuffle_order", "abl_old_ranker", "ours_2stage"]
SWEEP_BASES = ("repomap_sig", "ours", "snippet_rag")
SWEEP_BUDGETS = (512, 1024, 4096)
SWEEP = [f"{n}@{b}" for n in SWEEP_BASES for b in SWEEP_BUDGETS]
GROUPS = {"main": MAIN, "ablate": ABLATE, "sweep": SWEEP}

def registry(B=2048, ranker="hybrid"):
    base = dict(budget=B, ranker=ranker)
    M = {}
    def add(m): M[m.name] = m
    add(Method("issue_only", "issue_only", budget=0))
    add(Method("bm25_files", "bm25", budget=0))
    add(Method("raw_topk", "raw_topk", budget=B))
    add(Method("snippet_rag", "snippet_rag", budget=B))
    add(Method("repomap_sig", "graph", edges=False, control=False, **base))       # Aider-style (approximate re-implementation)
    add(Method("sig_edges", "graph", edges=True, control=False, **base))
    add(Method("sig_control", "graph", edges=False, control=True, **base))
    add(Method("ours", "graph", **base))
    add(Method("ours_2stage", "graph", two_stage=True, **base))
    add(Method("abl_json", "graph", fmt="json", **base))
    add(Method("abl_alpha_order", "graph", file_order="alpha", **base))
    add(Method("abl_shuffle_order", "graph", file_order="shuffle", **base))
    add(Method("abl_old_ranker", "graph", edges=False, control=False, budget=B, ranker="ppr"))
    for b in SWEEP_BUDGETS:
        for nm in SWEEP_BASES:
            add(replace(M[nm], name=f"{nm}@{b}", budget=b))
    return M
