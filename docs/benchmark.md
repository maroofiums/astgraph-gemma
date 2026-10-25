# Benchmark protocol and full results

## Protocol

- Data: SWE-bench Lite test split (300 instances, 12 repos), sorted by id, shuffled with seed 0. Dev = last 60. Test = first 150 of the rest. Ablations and the budget sweep use the first 60 test instances.
- Gold: files and functions touched by the reference patch. A function counts as gold only if every touched line is inside a function or class (119/150 test, 48/60 ablation set).
- Model: `google/gemma-4-E4B-it`, bitsandbytes NF4, fp16 compute, T4, SDPA, greedy.
- Dev split is used **only** to choose the ranker, with no LLM calls (`bench/diag.py`).
- Statistics: percentile bootstrap CIs (10k resamples); paired bootstrap for differences; Holm adjustment within each report (`bench/report.py`).
- Everything is resumable: `results/run_v3.jsonl` holds one row per (instance, method).

## Main table (n = 150, budget 2048)

| method | File@1 [95% CI] | File@3 | Func@1 [95% CI] | prompt tok | p50 s | p95 s |
|---|---|---|---|---|---|---|
| issue_only | 0.207 [.147,.273] | 0.253 | 0.109 [.059,.168] | 507 | 12.3 | 23.2 |
| bm25_files (no LLM) | 0.267 [.200,.340] | 0.460 | n/a | 0 | 4.7 | 15.9 |
| raw_topk | 0.280 [.213,.353] | 0.293 | 0.109 [.059,.168] | 2548 | 13.4 | 25.1 |
| snippet_rag | 0.380 [.300,.460] | 0.447 | 0.227 [.160,.303] | 2558 | 13.8 | 26.2 |
| ours (edges + control) | 0.487 [.407,.567] | 0.627 | 0.143 [.084,.210] | 2592 | 17.6 | 28.2 |
| repomap_sig | **0.527** [.447,.607] | **0.647** | **0.235** [.160,.311] | 2595 | 17.0 | 29.8 |

Paired differences against `repomap_sig` (Holm-adjusted p): bm25_files -0.260 (<.001); issue_only -0.320 (<.001) file, -0.126 (<.001) func; raw_topk -0.247 (<.001), -0.126 (.010); snippet_rag file -0.147 (.004), func -0.008 (.93); ours file -0.040 (.26), func -0.092 (.002).
Post hoc, unadjusted: ours vs snippet_rag file +0.107 (p = .023), func -0.084 (p = .088).

## Ablations (n = 60 / 48)

| variant | File@1 | File@3 | Func@1 | prompt tok |
|---|---|---|---|---|
| repomap_sig | 0.500 | 0.650 | 0.188 | 2614 |
| + edges | 0.467 | 0.667 | 0.125 | 2614 |
| + control flow | 0.533 | 0.683 | 0.208 | 2614 |
| ours (both) | 0.417 | 0.617 | 0.042 | 2612 |
| repomap_sig with PPR ranker | 0.333 | 0.483 | 0.062 | 2612 |
| ours, JSON rows | 0.400 | 0.633 | 0.062 | 2597 |
| ours, files alphabetical | 0.550 | 0.650 | 0.125 | 2605 |
| ours, files shuffled | 0.467 | 0.650 | 0.146 | 2605 |
| ours, two-stage | 0.467 | 0.617 | 0.229 | 3803 |

Only two Holm-adjusted differences are below 0.01: ours vs repomap_sig on function accuracy (worse), and two-stage vs ours on function accuracy (better). Everything else is inside the noise at this n. The first 60 instances are harder than the full 150 (ours scores 0.417 there, 0.487 on all 150).

## Budget sweep (n = 60)

| method | budget | File@1 | Func@1 | prompt tok |
|---|---|---|---|---|
| repomap_sig | 512 / 1024 / 2048 / 4096 | .533 / .550 / .500 / .517 | .167 / .188 / .188 / .188 | 1081 / 1588 / 2614 / 4662 |
| ours | 512 / 1024 / 2048 / 4096 | .467 / .483 / .417 / .583 | .125 / .146 / .042 / .167 | 1079 / 1591 / 2612 / 4663 |
| snippet_rag | 512 / 1024 / 2048 / 4096 | .383 / .433 / .433 / .500 | .208 / .229 / .271 / .313 | 1043 / 1553 / 2577 / 4598 |

## Failure analysis (first 60 test instances)

Gold file in prompt: ours 47/60, repomap_sig 48/60. Correct when present: 0.53 and 0.62. Correct when absent: 0/13 and 0/12. Typical model misses pick a sibling file (`django/db/models/query.py` vs `django/db/models/sql/query.py`).

## Corrections to the first notebook

The first version of the Kaggle notebook (4 hand-written issues on a toy repo) made claims that the v3 measurements do not support. They were removed from the paper and the updated notebook:

| first notebook said | v3 shows |
|---|---|
| peak VRAM under 2.34 GB | weights alone are 9.08 GiB; peak per call in the main runs averages about 10.1 GB (max 10.8 GB). The 2.34 GB figure cannot be reproduced from the v3 setup; the old run was not documented well enough to say why. |
| under 20 s per task | p50 is about 17 s on the main runs, p95 about 28 to 30 s, including repo ingest (median ingest 4.7 s, p95 15.9 s). |
| "prevented hallucinations, robust code navigation" | the gold file is missing from the prompt for about 20% of issues, and the model still misses the file 38 to 47% of the time when it is present. |
| structured JSON output was forced | decoding is plain greedy with a "JSON only" instruction; every one of the 750 LLM answers in the main run contained a parsable file list. |
| bf16 model | 4-bit NF4 with bf16 per-layer embeddings kept as is. |
