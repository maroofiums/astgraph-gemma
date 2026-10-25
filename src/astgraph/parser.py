from dataclasses import dataclass, field
from pathlib import Path
import re, hashlib
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY = Language(tspython.language())
_parser = Parser(PY)

CONTROL = {
    "if_statement": "if", "for_statement": "for", "while_statement": "while",
    "try_statement": "try", "with_statement": "with", "match_statement": "match",
}

@dataclass
class Symbol:
    qualname: str
    kind: str                      # class | function | method
    file: str
    start: int                     # 1-indexed line
    end: int
    signature: str
    calls: set = field(default_factory=set)
    control: dict = field(default_factory=dict)
    bases: list = field(default_factory=list)

def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_file(path: Path, root: Path, src=None) -> list[Symbol]:
    if src is None:
        src = path.read_bytes()
    tree = _parser.parse(src)
    rel = str(path.relative_to(root))
    out: list[Symbol] = []

    def text(n):
        return src[n.start_byte:n.end_byte].decode("utf8", "replace")

    def visit(node, scope, cur):
        t = node.type
        if t in ("function_definition", "class_definition"):
            name = text(node.child_by_field_name("name"))
            in_class = bool(scope) and scope[-1][1] == "class"
            if t == "class_definition":
                kind = "class"
                sup = node.child_by_field_name("superclasses")
                bases = [b.strip() for b in text(sup)[1:-1].split(",") if b.strip()] if sup else []
                sig = f"class {name}"
            else:
                kind = "method" if in_class else "function"
                params = text(node.child_by_field_name("parameters"))
                ret = node.child_by_field_name("return_type")
                sig = f"def {name}{params}" + (f" -> {text(ret)}" if ret else "")
                bases = []
            qual = ".".join([s[0] for s in scope] + [name])
            sym = Symbol(qual, kind, rel, node.start_point[0] + 1,
                         node.end_point[0] + 1, _squash(sig), bases=bases)
            out.append(sym)
            for ch in node.children:
                visit(ch, scope + [(name, kind)], sym)
            return
        if cur is not None:
            if t in CONTROL:
                k = CONTROL[t]
                cur.control[k] = cur.control.get(k, 0) + 1
            elif t == "call":
                fn = node.child_by_field_name("function")
                if fn is not None:
                    if fn.type == "attribute":
                        attr = fn.child_by_field_name("attribute")
                        cur.calls.add(text(attr))
                    elif fn.type == "identifier":
                        cur.calls.add(text(fn))
        for ch in node.children:
            visit(ch, scope, cur)

    visit(tree.root_node, [], None)
    return out

def extract_repo(root: Path, exclude=("venv", ".git", "node_modules", "__pycache__", ".tox"), cache=None):
    """cache: optional dict keyed by (relative path, sha1 of content); unchanged files are never re-parsed."""
    root = Path(root)
    syms = []
    for p in sorted(root.rglob("*.py")):
        if any(x in p.parts for x in exclude):
            continue
        try:
            src = p.read_bytes()
            if cache is not None:
                key = (str(p.relative_to(root)), hashlib.sha1(src).hexdigest())
                if key not in cache:
                    cache[key] = extract_file(p, root, src)
                syms.extend(cache[key])
            else:
                syms.extend(extract_file(p, root, src))
        except Exception:
            continue
    return syms
