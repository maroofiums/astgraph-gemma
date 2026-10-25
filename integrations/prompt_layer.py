"""Mode 1: prompt layer. The map is built before the model call and prepended to the prompt.

This is exactly the setup the benchmark measured. `generate` is any callable (system, user) -> text,
so it works with transformers, Ollama, llama.cpp, vLLM or a hosted API.

    python integrations/prompt_layer.py /path/to/repo "TypeError in Parser.parse_date when input is empty"
"""
import sys
from astgraph import RepoIndex

SYSTEM = ("You localize the cause of a reported issue in a Python repository. "
          'Return ONLY JSON: {"files": [up to 3 repo-relative paths, most likely first], '
          '"symbol": "qualified name (Class.method or function) of the single most likely location"}. '
          "Map legend: file path on its own line; indented rows are symbols. "
          "C class, m method, f function, Lx-y line range. Files are listed most relevant first.")


def localize(index: RepoIndex, issue: str, generate, budget: int = 2048) -> str:
    ctx = index.context(issue, budget=budget)
    return generate(SYSTEM, f"REPOSITORY MAP\n{ctx.text}\n\nISSUE\n{issue}")


def hf_generate(model_id="google/gemma-4-E4B-it", load_4bit=True):
    """Transformers backend, same loading code as the benchmark notebook."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(device_map={"": 0})
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()

    def generate(system, user, max_tokens=200):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate, tok


if __name__ == "__main__":
    repo, issue = sys.argv[1], sys.argv[2]
    gen, tok = hf_generate()
    from astgraph import hf_counter
    idx = RepoIndex.from_path(repo, count=hf_counter(tok))
    print(localize(idx, issue, gen))
