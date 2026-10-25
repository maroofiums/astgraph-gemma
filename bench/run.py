import json, re

LEGEND_COMPACT = ("Map legend: file path on its own line; indented rows are symbols. "
                  "C class, m method, f function, Lx-y line range, >callees, #control-flow counts.")
LEGEND_JSON = 'Map legend: one JSON row per file; n signature, l line range, c callees, f control-flow counts.'
SYS = ("You localize the cause of a reported issue in a Python repository. "
       'Return ONLY JSON: {"files": [up to 3 repo-relative paths, most likely first], '
       '"symbol": "qualified name (Class.method or function) of the single most likely location"}. ')
SYS_STAGE1 = ('You are shown repository files with their symbol names. Return ONLY JSON: '
              '{"files": [up to 3 repo-relative paths, most likely first]}.')
SNIP_NOTE = "The prompt lists code snippets; each starts with '# path:qualified_name Lstart-end'."

def clean_symbol(sym, files):
    s = (sym or "").strip()
    s = re.sub(r"^(?:[Cmf]\s+)?(?:def|class)\s+", "", s)
    for f in sorted(files, key=len, reverse=True):
        if f and s.startswith(f):
            s = s[len(f):].lstrip(".:/ ")
            break
    s = re.sub(r"\(.*$", "", s).strip()
    if ":" in s:
        s = s.split(":")[-1].strip()
    return s

def parse_pred(raw):
    try:
        d = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        files = [re.sub(r"^\./", "", f.strip()) for f in d.get("files", []) if isinstance(f, str)]
        return files, clean_symbol(d.get("symbol") or "", files)
    except Exception:
        return [], ""

def call(llm, system, user, sampler_on):
    if sampler_on:
        from .backends import VramSampler
        with VramSampler() as vs:
            text, st = llm.generate(system, user)
        st["vram_peak_mb"] = vs.peak
    else:
        text, st = llm.generate(system, user); st.setdefault("vram_peak_mb", None)
    return text, st

def run_method(m, ctx, llm, count, issue, G, scores, repo_dir, file_rank, vram):
    """ctx: dict of per-instance helpers ({'snip': SnippetIndex}). Returns pred_files, pred_symbol, stats, context_tokens."""
    from .contexts import graph_context, raw_context
    ctx = ctx or {}
    zero = dict(prompt_tokens=0, gen_tokens=0, prefill_ms=0, decode_ms=0, total_ms=0, wall_ms=0, vram_peak_mb=None)
    if m.kind == "bm25":
        return file_rank[:5], "", zero, 0
    if m.kind == "issue_only":
        text, st = call(llm, SYS, f"ISSUE\n{issue}", vram)
        f, s = parse_pred(text); return f, s, st, 0
    if m.kind == "raw_topk":
        body, ct = raw_context(repo_dir, file_rank[:10], count, m.budget)
        text, st = call(llm, SYS, f"REPOSITORY FILES\n{body}\n\nISSUE\n{issue}", vram)
        f, s = parse_pred(text); return f, s, st, ct
    if m.kind == "snippet_rag":
        body, ct, _ = ctx["snip"].select(issue, count, m.budget)
        text, st = call(llm, SYS + SNIP_NOTE, f"CODE SNIPPETS\n{body}\n\nISSUE\n{issue}", vram)
        f, s = parse_pred(text); return f, s, st, ct
    legend = (LEGEND_JSON if m.fmt == "json" else LEGEND_COMPACT)
    if m.file_order == "relevance":
        legend += " Files are listed most relevant first."
    if not m.two_stage:
        body, ct, _ = graph_context(G, scores, m, count)
        text, st = call(llm, SYS + legend, f"REPOSITORY MAP\n{body}\n\nISSUE\n{issue}", vram)
        f, s = parse_pred(text); return f, s, st, ct
    b1, c1, _ = graph_context(G, scores, m, count, budget=m.budget // 2, level="names")
    t1, st1 = call(llm, SYS_STAGE1, f"REPOSITORY MAP\n{b1}\n\nISSUE\n{issue}", vram)
    cand, _ = parse_pred(t1)
    cand = cand or None
    b2, c2, _ = graph_context(G, scores, m, count, budget=m.budget, restrict_files=set(cand) if cand else None)
    t2, st2 = call(llm, SYS + legend, f"REPOSITORY MAP\n{b2}\n\nISSUE\n{issue}", vram)
    f, s = parse_pred(t2)
    MAXK = ("vram_peak_mb", "weights_mb", "kv_act_mb")
    st = {k: (max(filter(None, [st1[k], st2.get(k)]), default=None) if k in MAXK
              else (st1[k] or 0) + (st2.get(k) or 0)) for k in st1}
    return f, s, st, c1 + c2
