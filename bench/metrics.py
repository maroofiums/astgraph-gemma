import numpy as np

def _last(q): return q.split(".")[-1]

def func_hit(pred_file, pred_sym, gold_funcs):
    if not pred_file or not pred_sym or pred_file not in gold_funcs:
        return False
    p = pred_sym.strip().replace("()", "")
    for g in gold_funcs[pred_file]:
        if g == "<module>":
            continue
        if p == g or _last(p) == _last(g) or p.startswith(g + "."):
            return True
    return False

def file_hits_at(pred_files, gold_files, ks=(1, 3, 5)):
    return {k: any(f in gold_files for f in pred_files[:k]) for k in ks}

def bootstrap_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])

def paired_bootstrap(a, b, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean(), lo, hi, float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))

def kv_cache_bytes(layer_groups, n_kv, d_head, T, bytes_per=2):
    """layer_groups: [(n_layers, window_or_None)]. Local layers cap cached tokens at their window.
    Read these from the model's config.json; do not assume them."""
    total = 0
    for n_layers, window in layer_groups:
        t_eff = T if window is None else min(T, window)
        total += 2 * n_layers * n_kv * d_head * t_eff * bytes_per
    return total
