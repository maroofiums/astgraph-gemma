# Method

## 1. Symbols

For each `.py` file (skipping `venv`, `.git`, `node_modules`, `__pycache__`, `.tox`) tree-sitter gives a tree. Every `class_definition` and `function_definition` becomes a `Symbol`. A function inside a class is a `method`; nested functions are recorded with a dotted qualified name. For each symbol we also collect:

- `signature`: `def name(params) -> ret` or `class Name`, whitespace squashed.
- `calls`: names at call sites (`f(x)` gives `f`, `obj.m(x)` gives `m`). Only the leaf name is kept.
- `control`: counts of `if/for/while/try/with/match` nodes inside the body.
- `bases`: base classes from the class header.

Symbol line ranges are 1-indexed and inclusive.

## 2. Graph

`calls` edges go from caller to every symbol with the same leaf name, weight `1/#candidates`. `inherits` edges go from a class to classes with the base's leaf name. There is no import resolution and no type inference, which is why `callers`/`callees` are labeled approximate.

## 3. Ranking

Each symbol is a bag of words: split identifiers from its qualified name, its file path and its callees (`parseDate_utc` becomes `parse date utc`). BM25 (k1=1.5, b=0.75, positive IDF) scores those documents against the identifier-split issue text.

| ranker | definition |
|---|---|
| `bm25` | BM25 score |
| `ppr` | personalised PageRank (alpha 0.85) on the undirected graph, personalisation = normalised BM25 |
| `bm25m` | BM25 / max, plus +1.0 if the issue names the symbol's file (path, or path in a traceback frame), +0.6 if it names the symbol's leaf (backticks, snake_case, CamelCase, traceback frame), +0.3 if it names an enclosing class |
| `hybrid` (default) | 0.7 * bm25m/max + 0.3 * PPR/max, PPR with alpha 0.5 seeded by the boosted scores |

The ranker was chosen on the dev split: the score is the mean of "gold file in prompt" and "gold function in prompt" at the 1024 and 2048 budgets for the signature map (hybrid 0.998, bm25m 0.973, bm25 0.889, ppr 0.443). Weights 0.7/0.3 and alpha 0.5 were set by hand, not tuned.

## 4. Budgeted rendering

1. Take the top 3000 symbols by score.
2. Cost of a symbol = tokens of its rendered row, plus the tokens of its file path if the file has not been used yet.
3. Sort by score / cost, take greedily while the total stays under the budget.
4. Group by file, order files by best score (`file_order="relevance"`), order rows by line number.
5. Count the final text and drop the last picked items while it is over budget. The budget is a hard limit.

Row format (compact): `  m def name(args) -> ret L10-42`, then optional `>callee1,callee2` (up to 8, sorted) and `#for2 if1`.
JSON format (one row per file) exists for ablation only.

## 5. Prompts used in the benchmark

System: "You localize the cause of a reported issue in a Python repository. Return ONLY JSON: {"files": [up to 3 repo-relative paths, most likely first], "symbol": "qualified name ..."}" plus a one-line legend for the map. User: `REPOSITORY MAP`, the map, `ISSUE`, the issue (truncated to 1024 tokens for every method). Greedy decoding, 200 new tokens max. The full text is in `bench/run.py`.

The two-stage variant first shows names only (half the budget), asks for candidate files, then re-renders the full map restricted to those files.
