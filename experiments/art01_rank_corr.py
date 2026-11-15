"""Artigo 1: correlação de postos (Spearman) entre macro de validação e de teste nas 11 configurações q_lerp_t. Saída: results/art01_rank_corr.txt"""
import os, numpy as np
from scipy.stats import spearmanr, kendalltau
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]
def mac(sp, t):
    z = np.load(os.path.join(ROOT, "cache", f"q_lerp_{t}__{sp}.npz")); return float(np.mean([z[k].mean() for k in TASKS]))
ts = [f"{t:.1f}" for t in np.arange(0, 1.01, .1)]
v = [mac("val", t) for t in ts]; s = [mac("test", t) for t in ts]
r, p = spearmanr(v, s); k, pk = kendalltau(v, s)
L = [f"t val_macro test_macro"] + [f"{t} {a:.4f} {b:.4f}" for t, a, b in zip(ts, v, s)]
L += [f"Spearman rho={r:.3f} p={p:.3f} (n=11)", f"Kendall tau={k:.3f} p={pk:.3f}"]
open(os.path.join(ROOT, "results", "art01_rank_corr.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
