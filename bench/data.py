import random, re
from collections import defaultdict

def load_instances(n=None, seed=0, repos=None, dataset="princeton-nlp/SWE-bench_Lite", split="test"):
    from datasets import load_dataset
    rows = [dict(r) for r in load_dataset(dataset, split=split)]
    if repos:
        rows = [r for r in rows if r["repo"] in repos]
    rows.sort(key=lambda r: r["instance_id"])
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows

def parse_patch(patch: str) -> dict:
    """Return {file: [old-side line numbers touched]}. Pure insertions map to the preceding old line."""
    out, cur, lines, i = {}, None, patch.splitlines(), 0
    def rec(ln):
        if cur is not None:
            out.setdefault(cur, []).append(max(ln, 1))
    while i < len(lines):
        l = lines[i]
        if l.startswith("--- "):
            p = l[4:].split("\t")[0].strip()
            cur = None if p == "/dev/null" else (p[2:] if p.startswith("a/") else p)
            i += 1
            continue
        if l.startswith("+++ "):
            i += 1
            continue
        if l.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", l)
            if not m:
                i += 1
                continue
            old = int(m[1]); ol = int(m[2]) if m[2] is not None else 1
            nl = int(m[4]) if m[4] is not None else 1
            i += 1
            while i < len(lines) and (ol > 0 or nl > 0):
                h = lines[i]
                if h.startswith("-"):
                    rec(old); old += 1; ol -= 1
                elif h.startswith("+"):
                    rec(old - 1); nl -= 1
                elif h.startswith("\\"):
                    pass
                else:
                    old += 1; ol -= 1; nl -= 1
                i += 1
            continue
        i += 1
    return out

def gold_from_patch(patch: str, symbols):
    """Return (gold_files, gold_funcs {file: set(qualname)}, funcs_defined: bool)."""
    touched = parse_patch(patch)
    by_file = defaultdict(list)
    for s in symbols:
        by_file[s.file].append(s)
    funcs = {}
    for f, lns in touched.items():
        names = set()
        for ln in lns:
            best = None
            for s in by_file.get(f, []):
                if s.start <= ln <= s.end and (best is None or s.end - s.start < best.end - best.start):
                    best = s
            names.add(best.qualname if best else "<module>")
        funcs[f] = names
    files = {f for f in touched if f in by_file or True}
    defined = bool(funcs) and all("<module>" not in v for v in funcs.values())
    return files, funcs, defined
