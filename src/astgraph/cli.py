import argparse, json, sys
from pathlib import Path
from .index import RepoIndex
from .text import hf_counter, estimate_tokens


def main(argv=None):
    ap = argparse.ArgumentParser(prog="astgraph", description="Budgeted AST repository maps for small local models.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="print a budgeted map for an issue")
    m.add_argument("repo"); m.add_argument("--issue", required=True, help="issue text, or @file to read it from a file")
    m.add_argument("--budget", type=int, default=2048)
    m.add_argument("--ranker", default="hybrid", choices=["bm25", "ppr", "bm25m", "hybrid"])
    m.add_argument("--edges", action="store_true"); m.add_argument("--control", action="store_true")
    m.add_argument("--tokenizer", default=None, help="HF tokenizer id for exact budgets (default: ~3.5 chars/token)")
    m.add_argument("--json", action="store_true", help="emit {text, tokens, files}")
    s = sub.add_parser("stats", help="symbol and edge counts")
    s.add_argument("repo")
    a = ap.parse_args(argv)
    count = hf_counter(a.tokenizer) if getattr(a, "tokenizer", None) else estimate_tokens
    idx = RepoIndex.from_path(Path(a.repo), count)
    if a.cmd == "stats":
        print(json.dumps({"symbols": len(idx.symbols), "files": len({x.file for x in idx.symbols}),
                          "edges": idx.graph.number_of_edges()}))
        return 0
    issue = Path(a.issue[1:]).read_text() if a.issue.startswith("@") else a.issue
    r = idx.context(issue, budget=a.budget, ranker=a.ranker, edges=a.edges, control=a.control)
    if a.json:
        print(json.dumps({"text": r.text, "tokens": r.tokens, "files": r.files}, indent=2))
    else:
        print(r.text); print(f"\n[{r.tokens} tokens, {len(r.files)} files]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
