# Memory on a T4

Everything here is measured with `HFLLM` in `notebooks/benchmark_v3.ipynb` (torch allocator peaks, 4-bit NF4, fp16 compute, T4 with 14.56 GiB usable). Raw numbers: `results/memory.csv`.

## Weights

| | MiB |
|---|---|
| resident after load | 9081 |
| parameters, bf16 | 6688 |
| parameters, uint8 (quantized) | 2115 |
| per-layer embeddings (`embed_tokens_per_layer`), bf16 | 5376 |
| `embed_tokens`, bf16 | 1280 |
| everything else ("compute weights"), the part that gets quantized | 2146 |

So "4-bit" shrinks the compute weights, not the embedding tables that make up most of this checkpoint.

## Architecture facts used for the analytic KV estimate

42 layers; 35 sliding-attention (window 512), 7 full-attention; 8 attention heads, 2 KV heads; head_dim 256 (35 layers) and 512 (7 layers); 18 layers share KV with earlier layers (`num_kv_shared_layers`); vocab 262144; context up to 131072. Analytic KV = sum over the 24 layers that own a cache of `2 * kv_heads * head_dim * min(T, window) * 2 bytes`.

## Prefill, measured vs analytic

| T | peak MiB | above weights | analytic KV MiB | KV share | prefill ms |
|---|---|---|---|---|---|
| 256 | 9182 | 93 | 14 | 15% | 304 |
| 512 | 9226 | 137 | 28 | 20% | 511 |
| 1024 | 9377 | 288 | 36 | 13% | 1009 |
| 2048 | 9816 | 727 | 52 | 7% | 2430 |
| 3072 | 10409 | 1320 | 68 | 5% | 4631 |
| 4096 | 11158 | 2069 | 84 | 4% | 9012 |
| 6144 | 13115 | 4026 | 116 | 3% | 13161 |
| 8192 | out of memory | | | | |

## Where the rest goes: logits

Same model, random token ids, extra MiB above weights:

| T | all-position logits | last-token logits | chunked 512 + last-token logits |
|---|---|---|---|
| 2048 | 2172 | 728 | 347 |
| 4096 | 4340 | 2069 | 624 |
| 6144 | OOM | 4026 | 909 |
| 8192 | OOM | OOM | 1191 (15.4 s) |

With a 262k vocabulary, logits for every position are the single biggest allocation. In Transformers: pass `logits_to_keep=1` and prefill in chunks with a `DynamicCache`. See the `probe` function in the notebook (section 8) for the exact code.

## What this does and does not say

- It says the context budget matters for prefill time (9 s at 4k tokens) and activation memory, and that KV cache is not the bottleneck for this architecture.
- It does not cover KV-cache quantization, other quantizations, other GPUs, or other runtimes (llama.cpp, vLLM, Ollama), which manage memory differently.
- The analytic KV number ignores shared-KV details beyond the layer count and any runtime padding.
