import json, time
from pathlib import Path
from astgraph.parser import extract_repo
from astgraph.graph import build_graph
from .contexts import rank_nodes, file_bm25, SnippetIndex
from .data import gold_from_patch
from .metrics import func_hit, file_hits_at
from .repos import checkout
from .run import run_method

def run_suite(llm, count, trunc_issue, methods, instances, out_path, checkout_fn=checkout, vram=False, verbose=True):
    """Resumable: skips (instance, method) pairs already present in out_path."""
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.open():
            r = json.loads(l); done.add((r["instance_id"], r["method"]))
    pcache = {}
    for k, inst in enumerate(instances):
        todo = [m for m in methods if (inst["instance_id"], m.name) not in done]
        if not todo:
            continue
        repo_dir = checkout_fn(inst["repo"], inst["base_commit"])
        t0 = time.perf_counter()
        syms = extract_repo(Path(repo_dir), cache=pcache); G = build_graph(syms)
        ingest_ms = (time.perf_counter() - t0) * 1000
        gold_files, gold_funcs, funcs_defined = gold_from_patch(inst["patch"], syms)
        issue = trunc_issue(inst["problem_statement"])
        file_rank = file_bm25(repo_dir, issue, sorted({s.file for s in syms}))
        ctx = {"snip": SnippetIndex(repo_dir, syms)} if any(m.kind == "snippet_rag" for m in todo) else {}
        scores = {}
        for m in todo:
            if m.kind == "graph" and m.ranker not in scores:
                scores[m.ranker] = rank_nodes(G, issue, m.ranker)
            t1 = time.perf_counter()
            pf, ps, st, ct = run_method(m, ctx, llm, count, issue, G, scores.get(m.ranker), repo_dir, file_rank, vram)
            hits = file_hits_at(pf, gold_files)
            rec = dict(instance_id=inst["instance_id"], repo=inst["repo"], method=m.name, ranker=m.ranker, budget=m.budget,
                       ctx_tokens=ct, ingest_ms=ingest_ms, n_symbols=len(syms),
                       file_acc1=hits[1], file_acc3=hits[3], file_acc5=hits[5],
                       func_acc1=func_hit(pf[0] if pf else None, ps, gold_funcs) if funcs_defined else None,
                       funcs_defined=funcs_defined, n_gold_files=len(gold_files), truncated=False,
                       pred_files=pf[:3], pred_symbol=ps, gold_files=sorted(gold_files),
                       gold_funcs={a: sorted(b) for a, b in gold_funcs.items()},
                       e2e_ms=ingest_ms + (time.perf_counter() - t1) * 1000, **st)
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        if verbose:
            print(f"[{k+1}/{len(instances)}]", inst["instance_id"], flush=True)
