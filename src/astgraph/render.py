"""Budgeted rendering of the symbol graph into a compact text map.

Greedy selection by score per token over the top-k ranked symbols. The rendered text is
guaranteed to fit the budget (checked with the supplied token counter).
"""
import json, random
from collections import defaultdict
from dataclasses import dataclass

@dataclass(frozen=True)
class Method:
    """Rendering options. Any object with these attributes works (bench.config.Method does too)."""
    fmt: str = "compact"            # compact | json
    edges: bool = False             # append callee names (>a,b)
    control: bool = False           # append control-flow counts (#if2 for1)
    budget: int = 2048
    file_order: str = "relevance"   # relevance | alpha | shuffle

# ---------------- graph rendering ----------------
def item_text(s, m: Method, level="full"):
    if level == "names":
        return s.qualname
    if m.fmt == "json":
        d = {"n": s.signature, "l": [s.start, s.end]}
        if m.edges and s.calls:
            d["c"] = sorted(s.calls)[:8]
        if m.control and s.control:
            d["f"] = s.control
        return json.dumps(d, separators=(",", ":"))
    tag = {"class": "C", "method": "m", "function": "f"}[s.kind]
    line = f"  {tag} {s.signature} L{s.start}-{s.end}"
    if m.edges and s.calls:
        line += " >" + ",".join(sorted(s.calls)[:8])
    if m.control and s.control:
        line += " #" + " ".join(f"{k}{v}" for k, v in sorted(s.control.items()))
    return line

def _cost(s, m, level, count):
    tc = getattr(s, "_tc", None)
    if tc is None:
        tc = {}; s._tc = tc
    key = (m.fmt, m.edges, m.control, level, id(count))
    if key not in tc:
        tc[key] = count(item_text(s, m, level))
    return tc[key]

def _file_order(picked, m):
    best = {}
    for f, _, _, sc in picked:
        best[f] = max(best.get(f, float("-inf")), sc)
    if m.file_order == "alpha":
        return sorted(best)
    if m.file_order == "shuffle":
        fs = sorted(best); random.Random(0).shuffle(fs); return fs
    return sorted(best, key=lambda f: (-best[f], f))

def _build(picked, m, level):
    by = defaultdict(list)
    for f, st, t, sc in picked:
        by[f].append((st, t))
    lines = []
    for f in _file_order(picked, m):
        items = [t for _, t in sorted(by[f])]
        if level == "names" or m.fmt != "json":
            lines.append(f)
            lines += (["  " + t for t in items] if level == "names" else items)
        else:
            lines.append(f'"{f}":[' + ",".join(items) + "]")
    return "\n".join(lines)

def graph_context(G, scores, m: Method, count, budget=None, restrict_files=None, level="full", topk=3000):
    """Greedy score/token selection over the top-k ranked symbols; rendered text is guaranteed <= budget tokens."""
    budget = m.budget if budget is None else budget
    cand = scores.items()
    if restrict_files is not None:
        cand = [(n, sc) for n, sc in cand if G.nodes[n]["sym"].file in restrict_files]
    cand = sorted(cand, key=lambda kv: -kv[1])[:topk]
    items = []
    for n, sc in cand:
        s = G.nodes[n]["sym"]
        tau = max(_cost(s, m, level, count), 1)
        items.append((sc / tau, sc, s, tau))
    items.sort(key=lambda x: -x[0])
    picked, used, seen, fcost = [], 0, set(), {}
    for _, sc, s, tau in items:
        fc = 0 if s.file in seen else fcost.setdefault(s.file, count(s.file))
        if used + tau + fc > budget:
            continue
        used += tau + fc; seen.add(s.file)
        picked.append((s.file, s.start, item_text(s, m, level), sc))
    text = _build(picked, m, level)
    while picked and count(text) > budget:
        picked.pop(); text = _build(picked, m, level)
    return text, (count(text) if text else 0), _file_order(picked, m)

