# Architecture

```mermaid
flowchart TB
    subgraph lib[src/astgraph]
        parser[parser.py<br/>tree-sitter to Symbol]
        sgraph[graph.py<br/>networkx DiGraph]
        ranking[ranking.py<br/>BM25, mentions, PPR]
        render[render.py<br/>budgeted fill]
        index[index.py<br/>RepoIndex]
        tools[tools.py<br/>ToolKit + specs]
        cli[cli.py]
    end
    parser --> sgraph --> index
    ranking --> index
    render --> index
    index --> tools
    index --> cli
    tools --> mcp[integrations/mcp_server.py]
    tools --> loop[integrations/tool_calling.py]
    index --> layer[integrations/prompt_layer.py]
    subgraph bench[bench/]
        data[data.py<br/>SWE-bench Lite, gold from patch]
        contexts[contexts.py<br/>baselines]
        run[run.py, suite.py<br/>prompts, resumable runs]
        diag[diag.py<br/>dev coverage, no LLM]
        report[report.py, metrics.py<br/>CIs, paired tests]
    end
    index -.used by.-> run
    contexts --> run
```

**Library vs benchmark.** `src/astgraph` is what you import in a project. `bench/` is the measurement harness: dataset loading, the baselines (BM25 over files, raw top-k files, function-snippet RAG), prompts, the resumable runner and the statistics. The benchmark imports the library; the library never imports the benchmark.

**Data model.** `Symbol` holds `qualname`, `kind` (`class|function|method`), `file`, `start`/`end` line, `signature`, `calls` (set of callee names), `control` (counts per statement kind) and `bases`. A node id is `qualname@file`. Edges are `calls` (weight 1/#candidates with that leaf name) and `inherits`.

**Caching.** `extract_repo(root, cache=dict)` keys parsed files by `(relative path, sha1)`. Re-indexing after a commit only parses changed files. The token cost of each rendered symbol is cached on the symbol, keyed by format, flags and the identity of the counter function, so use one counter object per index.

**Equivalence with the benchmark code.** The library was split out of the original v3 benchmark package (`contexts.py`). Before releasing, rendered maps and rankings were compared old vs new on the networkx source for 3 issues x 7 methods x 2 budgets plus all four rankers: identical text, file order and token counts (raw scores differ in the last floating-point digit between processes because of set iteration order).
