"""Small text helpers: identifier splitting and token counting."""
import re


def split_ident(s: str):
    """'parseDate_utc' -> ['parse', 'date', 'utc']"""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", s) if w]


def estimate_tokens(s: str) -> int:
    """Rough count (about 3.5 characters per token for code). Good enough for a first pass;
    pass a real tokenizer counter when the budget has to be exact."""
    return max(1, round(len(s) / 3.5))


def hf_counter(tokenizer_or_id):
    """Exact counter for a Hugging Face tokenizer (object or model id)."""
    tok = tokenizer_or_id
    if isinstance(tok, str):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok)
    return lambda s: len(tok.encode(s, add_special_tokens=False))
