import sys, tempfile, textwrap, json, random
from pathlib import Path
sys.path.insert(0, ".")
from astgraph.parser import extract_repo
from astgraph.graph import build_graph
from bench.data import parse_patch, gold_from_patch
from bench.config import registry, GROUPS
from bench.contexts import rank_nodes, graph_context, file_bm25, SnippetIndex, mentions
from bench.metrics import func_hit, bootstrap_ci, paired_bootstrap
from bench.backends import MockLLM
from bench.run import parse_pred
from bench.suite import run_suite
from bench.diag import coverage_for_instance

count = lambda s: max(1, round(len(s) / 3.5))
d = Path(tempfile.mkdtemp()); (d / "pkg").mkdir()
rnd = random.Random(1)
for i in range(40):
    (d / f"pkg/m{i}.py").write_text(textwrap.dedent(f'''
        class C{i}:
            def f{i}(self, x):
                if x:
                    return g{i}(x)
                for a in x:
                    helper_{i % 5}(a)
            def g{i}(self, y):
                try:
                    return y
                except ValueError:
                    return None
        def helper_{i % 5}(z):
            return z
    '''))
(d / "pkg/a.py").write_text(textwrap.dedent('''
    class Parser:
        def parse_date(self, s):
            if not s:
                return None
            return self.convert(s)
        def convert(self, s):
            return s
'''))
syms = extract_repo(d, cache={}); G = build_graph(syms)
issue = 'Traceback: File "/x/pkg/a.py", line 4, in parse_date\nParser.parse_date returns None for empty `s`'
paths, idents = mentions(issue); print("mentions:", paths, sorted(idents)[:6])
assert "parse_date" in idents
fr = file_bm25(d, issue, sorted({s.file for s in syms})); assert fr[0] == "pkg/a.py", fr[:3]
for rk in ("bm25", "ppr", "bm25m", "hybrid"):
    sc = rank_nodes(G, issue, rk)
    top = max(sc, key=sc.get); print(f"{rk:7s} top symbol:", top)
    assert "parse_date" in top or "Parser" in top or rk in ("ppr",)
reg = registry(512)
for B in (256, 512, 1024):
    for name in ("ours", "repomap_sig", "abl_json", "abl_alpha_order", "abl_shuffle_order"):
        m = registry(B)[name]
        body, ct, order = graph_context(G, rank_nodes(G, issue, m.ranker), m, count)
        assert ct <= B, (name, B, ct)
m = reg["ours"]; body, ct, order = graph_context(G, rank_nodes(G, issue, "hybrid"), m, count)
print(body[:300]); assert order[0] == "pkg/a.py", order[:3]
sn = SnippetIndex(d, syms); body, ct, files = sn.select(issue, count, 400); assert ct <= 400 and files[0] == "pkg/a.py", (ct, files[:2])
assert parse_pred('{"files":["./src/a.py"],"symbol":"src/a.py:Parser.parse_date()"}') == (["src/a.py"], "Parser.parse_date")
assert parse_pred('{"files":["x.py"],"symbol":"m def agg(self)"}')[1] == "agg"

patch = """--- a/pkg/a.py
+++ b/pkg/a.py
@@ -3,5 +3,5 @@
         def parse_date(self, s):
             if not s:
-                return None
+                return ''
             return self.convert(s)
         def convert(self, s):
"""
inst = dict(instance_id="t__1", repo="t/t", base_commit="x", patch=patch, problem_statement=issue)
insts = [dict(inst, instance_id=f"t__{i}") for i in range(3)]
out = Path(tempfile.mkdtemp()) / "run.jsonl"
names = GROUPS["main"] + GROUPS["ablate"] + GROUPS["sweep"]
R = registry(2048); methods = [R[n] for n in names]
run_suite(MockLLM(), count, lambda t: t, methods, insts, out, checkout_fn=lambda r, c: d, verbose=False)
rows = [json.loads(l) for l in out.open()]
assert len(rows) == 3 * len(names), (len(rows), len(names))
print("rows:", len(rows), "| example:", {k: rows[5][k] for k in ("method", "file_acc1", "func_acc1", "ctx_tokens")})
run_suite(MockLLM(), count, lambda t: t, methods, insts, out, checkout_fn=lambda r, c: d, verbose=False)
assert len(list(out.open())) == len(rows), "resume failed"
row = coverage_for_instance(inst, d, count, lambda t: t, pcache={})
assert row["repomap_sig|hybrid|512|file"] and row["snippet_rag||512|file"], row
print("coverage keys:", len(row))
import subprocess
r = subprocess.run([sys.executable, "-m", "bench.report", "--results", str(out), "--ref", "repomap_sig",
                    "--methods", "ours,repomap_sig,snippet_rag"], capture_output=True, text=True)
print(r.stdout[-900:], r.stderr[-300:]); assert r.returncode == 0
print(bootstrap_ci([1, 0, 1, 1] * 10)[0], paired_bootstrap([1, 1, 0, 1] * 10, [0, 1, 0, 0] * 10)[3] <= 1.0)
print("SMOKE OK")
