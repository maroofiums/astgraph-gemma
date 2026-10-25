import re, hashlib
from collections import defaultdict, Counter
from pathlib import Path
from astgraph.text import split_ident
from astgraph.ranking import CBM25, seed_scores, mentions, rank_nodes
from astgraph.render import graph_context, item_text
from .config import Method

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_SNIP_LINES = 40

# ---------------- file-level BM25 and raw baselines ----------------
_FTF = {}
def file_bm25(repo_dir, issue, files):
    tfs, kept = [], []
    for f in files:
        try:
            b = (Path(repo_dir) / f).read_bytes()
        except Exception:
            continue
        key = (f, hashlib.sha1(b).hexdigest())
        tf = _FTF.get(key)
        if tf is None:
            body = b.decode("utf8", "replace")
            tf = Counter(w for t in _TOK.findall(body) for w in split_ident(t))
            tf.update(split_ident(f))
            _FTF[key] = tf
        tfs.append(tf); kept.append(f)
    if not tfs:
        return []
    sc = CBM25(tfs).scores(split_ident(issue))
    return [f for f, _ in sorted(zip(kept, sc), key=lambda x: -x[1])]

def raw_context(repo_dir, ranked_files, count, budget):
    parts, used = [], 0
    for f in ranked_files:
        body = (Path(repo_dir) / f).read_text(errors="replace")
        chunk = f"### {f}\n{body}\n"
        c = count(chunk)
        if used + c > budget:
            frac = max(budget - used, 0) / c
            parts.append(chunk[: int(len(chunk) * frac)]); used = budget
            break
        parts.append(chunk); used += c
    text = "".join(parts)
    return text, count(text) if text else 0

# ---------------- function-level snippet RAG baseline ----------------
_SNIPS = {}
class SnippetIndex:
    """BM25 over function/method bodies (first 40 lines each) - the standard 'retrieve code chunks' baseline."""
    def __init__(self, repo_dir, syms):
        self.chunks, tfs = [], []
        by_file = defaultdict(list)
        for s in syms:
            if s.kind != "class":
                by_file[s.file].append(s)
        for f, ss in by_file.items():
            try:
                b = (Path(repo_dir) / f).read_bytes()
            except Exception:
                continue
            key = (f, hashlib.sha1(b).hexdigest())
            ent = _SNIPS.get(key)
            if ent is None:
                lines = b.decode("utf8", "replace").splitlines()
                ent = []
                for s in ss:
                    body = "\n".join(lines[s.start - 1: min(s.end, s.start - 1 + _SNIP_LINES)])
                    tf = Counter(w for t in _TOK.findall(s.qualname + " " + body) for w in split_ident(t))
                    ent.append((s.qualname, s.start, s.end, tf, body))
                _SNIPS[key] = ent
            for q, st, en, tf, body in ent:
                self.chunks.append((f, q, st, en, body)); tfs.append(tf)
        self.bm = CBM25(tfs)

    def select(self, issue, count, budget):
        if not self.chunks:
            return "", 0, []
        sc = self.bm.scores(split_ident(issue))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:400]
        parts, used, files = [], 0, []
        for i in order:
            f, q, st, en, body = self.chunks[i]
            txt = f"# {f}:{q} L{st}-{en}\n{body}\n"
            c = count(txt)
            if used + c > budget:
                continue
            parts.append(txt); used += c
            if f not in files:
                files.append(f)
            if used > 0.97 * budget:
                break
        text = "".join(parts)
        while parts and count(text) > budget:
            parts.pop(); text = "".join(parts)
        return text, (count(text) if text else 0), files
