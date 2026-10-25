"""Mode 2: tool mode. The model decides when to look at the map and which symbols to read.

Status: the tool layer (astgraph.tools) is unit-tested, and this loop is tested with a scripted model.
It has NOT been benchmarked against Gemma 4 yet; see docs/integration.md for the experiment to run.

Two ways to get tool calls out of the model:
  * runtime with native function calling (Ollama, vLLM, llama.cpp server): pass `tools=TOOL_SPECS`
    and read `message.tool_calls`. `ollama_chat` below does that.
  * plain text: ask for one JSON object per turn and parse it with `parse_tool_call`.
"""
import json, re
from astgraph import RepoIndex
from astgraph.tools import ToolKit, TOOL_SPECS

AGENT_SYSTEM = (
    "You find where a reported issue in a Python repository should be fixed. You have tools: repo_map, "
    "find_symbol, read_symbol, callers, callees, outline. Start with repo_map. Read at most a few symbols. "
    'When done, answer with JSON only: {"files": [...up to 3 paths...], "symbol": "Class.method"}.')

_JSON = re.compile(r"\{.*\}", re.S)


def parse_tool_call(text):
    """Text fallback: {"tool": "read_symbol", "arguments": {...}} (or name/arguments) -> (name, args) or None."""
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    name = d.get("tool") or d.get("name")
    if name in {t["function"]["name"] for t in TOOL_SPECS}:
        return name, d.get("arguments") or d.get("args") or {}
    return None


def run_agent(index: RepoIndex, issue: str, chat, max_steps: int = 6):
    """chat(messages, tools) -> {"content": str, "tool_calls": [{"name":..,"arguments":..}, ...]}.
    Returns (final_text, transcript)."""
    kit = ToolKit(index)
    msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": f"ISSUE\n{issue}"}]
    for _ in range(max_steps):
        reply = chat(msgs, TOOL_SPECS)
        calls = reply.get("tool_calls") or []
        if not calls:
            pc = parse_tool_call(reply.get("content", ""))
            calls = [{"name": pc[0], "arguments": pc[1]}] if pc else []
        if not calls:
            return reply.get("content", ""), msgs
        msgs.append({"role": "assistant", "content": reply.get("content", ""), "tool_calls": calls})
        for c in calls:
            msgs.append({"role": "tool", "name": c["name"], "content": kit.dispatch(c["name"], c.get("arguments", {}))})
    msgs.append({"role": "user", "content": "Step limit reached. Answer now with the JSON."})
    return chat(msgs, None).get("content", ""), msgs


def ollama_chat(model="gemma4:e4b", host="http://localhost:11434"):
    """Adapter for Ollama's /api/chat with native tool calling. Model tag is a placeholder: use what you pulled."""
    import urllib.request

    def chat(messages, tools):
        payload = {"model": model, "stream": False, "messages": messages, "options": {"temperature": 0}}
        if tools:
            payload["tools"] = tools
        req = urllib.request.Request(host + "/api/chat", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            msg = json.loads(r.read())["message"]
        calls = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments", {})} for c in msg.get("tool_calls", [])]
        return {"content": msg.get("content", ""), "tool_calls": calls}
    return chat
