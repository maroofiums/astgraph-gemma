# Reproducing the numbers

## Unit tests (no GPU, no network)

```bash
pip install -e ".[dev,mcp]"
pytest
```

16 tests: parser, budget guarantee, ranking, parse cache, tools, CLI, agent loop with a scripted model, MCP registration, and an end-to-end benchmark smoke test with a mock LLM (resume, report, coverage diagnostic).

## Full benchmark

Kaggle (GPU T4, Internet on, secret `HF_TOKEN`) or Colab T4. Open `notebooks/benchmark_v3.ipynb` and run all cells. It embeds the benchmark code, so it does not need this repository. Expect 4 to 5 hours with the defaults (`N_TEST=150`, ablations 107 min, sweep 120 min); set `FULL=False` or lower `N_TEST` for a quick pass. Results append to `run_v3.jsonl`, so an interrupted run resumes.

The benchmark clones the SWE-bench repositories from GitHub at the base commit of each instance (`~/.cache/astgraph_bench`).

Note: the notebook embeds the original v3 package layout (`astgraph/` and `bench/contexts.py`). This repository splits `contexts.py` into library modules (`ranking.py`, `render.py`); behavior is identical (see `docs/architecture.md`). To run the harness from this repo instead:

```python
from bench.suite import run_suite   # same arguments as in the notebook
```

## Reports and figures from saved results

```bash
python -m bench.report --results results/run_v3.jsonl --ref repomap_sig \
    --methods issue_only,bm25_files,snippet_rag,raw_topk,repomap_sig,ours
python paper/make_figs.py            # writes paper/figs/*.pdf and docs/img/*.png
cd paper && latexmk -pdf main.tex
```

`bench.report` needs pandas and numpy (`requirements-bench.txt`).

## Files in results/

| file | what |
|---|---|
| `run_v3.jsonl` | one row per (instance, method): predictions, gold, tokens, timings, memory |
| `coverage_dev.csv` | dev split: gold file / function present in prompt, per ranker, budget, method |
| `sweep.csv` | budget sweep table |
| `memory.csv` | prefill memory and time by prompt length |
| `*.png` | the original notebook plots |
