import networkx as nx
from collections import defaultdict

def nid(s): return s.qualname + "@" + s.file

def build_graph(symbols):
    G = nx.DiGraph()
    by_name = defaultdict(list)
    for s in symbols:
        G.add_node(nid(s), sym=s)
        by_name[s.qualname.split(".")[-1]].append(s)
    for s in symbols:
        for callee in s.calls:
            cands = by_name.get(callee, [])
            for c in cands:
                if nid(c) != nid(s):
                    G.add_edge(nid(s), nid(c), kind="calls", w=1.0 / len(cands))
        for b in s.bases:
            for c in by_name.get(b.split(".")[-1], []):
                if nid(c) != nid(s):
                    G.add_edge(nid(s), nid(c), kind="inherits", w=1.0)
    return G
