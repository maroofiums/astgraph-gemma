"""Tool mode: expose the index to an LLM as a handful of function-calling tools.

The map is no longer pushed into the prompt. The model asks for it (repo_map), then reads
only the symbols it picks (read_symbol), and can follow call edges on demand (callers, callees).
Specs use the common JSON-schema function format, which Hugging Face chat templates
(`apply_chat_template(..., tools=specs)`), Ollama and OpenAI-style APIs accept.
"""
import json
from .index import RepoIndex

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "repo_map",
        "description": "Ranked map of the repository for a problem description: file paths with the signatures of the most relevant classes, methods and functions, within a token budget. Call this first.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Issue text or a short description of what to find."},
            "budget": {"type": "integer", "description": "Token budget for the map (default 1024)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "find_symbol",
        "description": "Look up a class, method or function by name. Accepts 'name', 'Class.method' or 'path.py::name'.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "read_symbol",
        "description": "Source code of one symbol (first max_lines lines).",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Qualified name, optionally 'path.py::Class.method'."},
            "max_lines": {"type": "integer"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "callers",
        "description": "Symbols that call the given symbol (name-based, approximate).",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "callees",
        "description": "Symbols called by the given symbol (name-based, approximate).",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "outline",
        "description": "All symbols defined in one file, with line ranges.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]


def _fmt(s):
    return f"{s.file}::{s.qualname} [{s.kind}] L{s.start}-{s.end}"


class ToolKit:
    def __init__(self, index: RepoIndex, max_chars: int = 4000, max_items: int = 15):
        self.idx, self.max_chars, self.max_items = index, max_chars, max_items

    def _cap(self, text: str) -> str:
        return text if len(text) <= self.max_chars else text[: self.max_chars] + "\n... (truncated)"

    def _one(self, name):
        hits = self.idx.find(name)
        if not hits:
            return None, f"No symbol matches '{name}'. Try repo_map or find_symbol with a different name."
        if len(hits) > 1:
            lines = "\n".join(_fmt(s) for s in hits[: self.max_items])
            return None, f"{len(hits)} symbols match '{name}'; pick one and use 'path.py::qualname':\n{lines}"
        return hits[0], ""

    def repo_map(self, query, budget=1024):
        r = self.idx.context(query, budget=int(budget))
        return self._cap(r.text or "(empty map)")

    def find_symbol(self, name):
        hits = self.idx.find(name)
        if not hits:
            return f"No symbol matches '{name}'."
        return self._cap("\n".join(_fmt(s) for s in hits[: self.max_items]))

    def read_symbol(self, name, max_lines=60):
        s, err = self._one(name)
        return err if s is None else self._cap(f"# {_fmt(s)}\n" + self.idx.source(s, int(max_lines)))

    def callers(self, name):
        s, err = self._one(name)
        if s is None:
            return err
        out = self.idx.callers(s)
        return self._cap("\n".join(_fmt(c) for c in out[: self.max_items]) or "no callers found")

    def callees(self, name):
        s, err = self._one(name)
        if s is None:
            return err
        out = self.idx.callees(s)
        return self._cap("\n".join(_fmt(c) for c in out[: self.max_items]) or "no callees found")

    def outline(self, path):
        syms = [s for s in self.idx.symbols if s.file == path or s.file.endswith("/" + path)]
        if not syms:
            return f"No file '{path}' in the index."
        return self._cap("\n".join(f"{'  ' if s.kind == 'method' else ''}{s.signature}  L{s.start}-{s.end}" for s in sorted(syms, key=lambda s: s.start)))

    def dispatch(self, name: str, arguments) -> str:
        """Run a tool call. `arguments` may be a dict or a JSON string. Never raises: errors come back as text."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            fn = getattr(self, name, None)
            if name.startswith("_") or name not in {t["function"]["name"] for t in TOOL_SPECS} or fn is None:
                return f"Unknown tool '{name}'."
            return fn(**args)
        except Exception as e:  # keep the agent loop alive
            return f"Tool error ({type(e).__name__}): {e}"
