"""RepoIndex: parse a repository once, then ask for budgeted maps or navigate the graph."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .graph import build_graph, nid
from .parser import extract_repo
from .ranking import rank_nodes
from .render import Method, graph_context
from .text import estimate_tokens


@dataclass
class ContextResult:
    text: str
    tokens: int
    files: list = field(default_factory=list)   # files in the map, most relevant first


class RepoIndex:
    """Holds the symbols and the call/inheritance graph of one checkout.

    idx = RepoIndex.from_path("path/to/repo")
    ctx = idx.context("TypeError in Parser.parse_date for empty input", budget=2048)
    prompt = ctx.text + "\\n\\nISSUE\\n..."
    """

    def __init__(self, root, symbols, count: Callable[[str], int] = estimate_tokens):
        self.root = Path(root)
        self.symbols = symbols
        self.graph = build_graph(symbols)
        self.count = count
        self._by_leaf = {}
        for s in symbols:
            self._by_leaf.setdefault(s.qualname.split(".")[-1], []).append(s)

    @classmethod
    def from_path(cls, root, count: Callable[[str], int] = estimate_tokens, cache: Optional[dict] = None):
        """cache: optional dict shared between calls; unchanged files (same sha1) are not re-parsed."""
        return cls(root, extract_repo(Path(root), cache=cache), count)

    # ---- the prompt-layer entry point -------------------------------------------------
    def rank(self, issue: str, ranker: str = "hybrid"):
        return rank_nodes(self.graph, issue, ranker)

    def context(self, issue: str, budget: int = 2048, ranker: str = "hybrid", edges: bool = False,
                control: bool = False, fmt: str = "compact", restrict_files=None, level: str = "full",
                scores=None) -> ContextResult:
        """Budgeted repository map for an issue. Defaults are the best configuration in the benchmark
        (signatures only, hybrid ranker). edges/control reproduce the 'ours' variant."""
        m = Method(fmt=fmt, edges=edges, control=control, budget=budget)
        sc = scores if scores is not None else self.rank(issue, ranker)
        text, tokens, files = graph_context(self.graph, sc, m, self.count, restrict_files=restrict_files, level=level)
        return ContextResult(text, tokens, files)

    # ---- navigation (used by tool mode) -------------------------------------------------
    def find(self, name: str):
        """Symbols whose qualified name equals `name` or ends with it (Class.method, method, or file::name)."""
        name = name.strip()
        file_hint = None
        if "::" in name:
            file_hint, name = name.split("::", 1)
        out = [s for s in self.symbols if s.qualname == name or s.qualname.endswith("." + name)]
        if file_hint:
            out = [s for s in out if s.file.endswith(file_hint)]
        return out

    def source(self, sym, max_lines: int = 80) -> str:
        lines = (self.root / sym.file).read_text(errors="replace").splitlines()
        body = lines[sym.start - 1: min(sym.end, sym.start - 1 + max_lines)]
        more = sym.end - sym.start + 1 - len(body)
        tail = f"\n# ... {more} more lines" if more > 0 else ""
        return "\n".join(body) + tail

    def callees(self, sym):
        k = nid(sym)
        return [self.graph.nodes[v]["sym"] for _, v, d in self.graph.out_edges(k, data=True) if d.get("kind") == "calls"]

    def callers(self, sym):
        k = nid(sym)
        return [self.graph.nodes[u]["sym"] for u, _, d in self.graph.in_edges(k, data=True) if d.get("kind") == "calls"]
