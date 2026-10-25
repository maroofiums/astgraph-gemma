from astgraph.text import estimate_tokens, hf_counter


def make_counter(tokenizer_id=None):
    """Exact counts if a HF tokenizer id is given, else a ~3.5 chars/token estimate."""
    return hf_counter(tokenizer_id) if tokenizer_id else estimate_tokens
