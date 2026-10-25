"""Symbol ranking: BM25 over symbol documents, an issue-mention boost, and personalised PageRank.

Four rankers are available: "bm25", "ppr", "bm25m" (BM25 + mention boost) and "hybrid"
(0.7 * boosted BM25 + 0.3 * PPR). "hybrid" is the default; it was chosen on a dev split
without calling the LLM (see docs/benchmark.md).
"""
import re
from collections import defaultdict, Counter
import math
import networkx as nx
from .text import split_ident

_PATH = re.compile(r"[\w./-]+\.py")
_TICK = re.compile(r"`([^`\n]{2,80})`")
_FRAME = re.compile(r'File "([^"]+\.py)", line \d+, in (\w+)')
_CAMEL = re.compile(r"[a-z0-9][A-Z]")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

class CBM25:
    """BM25 over term-frequency Counters with an inverted index (positive IDF)."""
    def __init__(self, tfs, k1=1.5, b=0.75):
        self.k1, self.b, self.N = k1, b, len(tfs)
        self.dl = [sum(t.values()) for t in tfs]
        self.avg = (sum(self.dl) / self.N) if self.N else 1.0
        self.post = defaultdict(list)
        for i, t in enumerate(tfs):
            for w, f in t.items():
                self.post[w].append((i, f))
    def scores(self, q):
        out = [0.0] * self.N
        k1, b, avg = self.k1, self.b, (self.avg or 1.0)
        for w in set(q):
            pl = self.post.get(w)
            if not pl:
                continue
            idf = math.log((self.N - len(pl) + 0.5) / (len(pl) + 0.5) + 1.0)
            for i, f in pl:
                out[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.dl[i] / avg))
        return out


# ---------------- symbol ranking ----------------
def _doc(s):
    d = getattr(s, "_doc", None)
    if d is None:
        d = Counter(split_ident(s.qualname) + split_ident(s.file) + [w for c in s.calls for w in split_ident(c)])
        s._doc = d
    return d

def seed_scores(G, issue):
    nodes = list(G.nodes)
    if not nodes:
        return {}
    bm = CBM25([_doc(G.nodes[n]["sym"]) for n in nodes])
    return dict(zip(nodes, bm.scores(split_ident(issue))))

def mentions(issue):
    paths, idents = set(), set()
    for p in _PATH.findall(issue):
        paths.add(p)
    for f, fn in _FRAME.findall(issue):
        paths.add(f); idents.add(fn)
    for t in _TICK.findall(issue):
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", t):
            if len(w) >= 3:
                idents.add(w)
    for w in _WORD.findall(issue):
        if "_" in w or _CAMEL.search(w):
            idents.add(w)
    return paths, idents

def _path_key(p):
    parts = [x for x in p.replace("\\", "/").split("/") if x and x != "."]
    return "/".join(parts[-2:])

def _boost(s, keys, idents):
    b = 0.0
    if keys and any(s.file == k or s.file.endswith("/" + k) or s.file.endswith(k) for k in keys):
        b += 1.0
    parts = s.qualname.split(".")
    if parts[-1] in idents:
        b += 0.6
    elif any(p in idents for p in parts):
        b += 0.3
    return b

def _und(G):
    if "_und" not in G.graph:
        G.graph["_und"] = G.to_undirected()
    return G.graph["_und"]

def rank_nodes(G, issue, ranker, alpha=0.85):
    seed = seed_scores(G, issue)
    if not seed or ranker == "bm25":
        return seed
    mx = max(seed.values()) or 1.0
    base = {n: max(v, 0.0) / mx for n, v in seed.items()}
    if ranker == "ppr":
        tot = sum(base.values())
        pers = {n: v / tot for n, v in base.items()} if tot > 0 else None
        return nx.pagerank(_und(G), alpha=alpha, personalization=pers, weight="w")
    paths, idents = mentions(issue)
    keys = {k for k in (_path_key(p) for p in paths) if k}
    boosted = {n: base[n] + _boost(G.nodes[n]["sym"], keys, idents) for n in base}
    if ranker == "bm25m":
        return boosted
    tot = sum(boosted.values()) or 1.0
    ppr = nx.pagerank(_und(G), alpha=0.5, personalization={n: v / tot for n, v in boosted.items()}, weight="w")
    bm, pm = (max(boosted.values()) or 1.0), (max(ppr.values()) or 1.0)
    return {n: 0.7 * boosted[n] / bm + 0.3 * ppr[n] / pm for n in boosted}

