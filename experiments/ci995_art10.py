"""IC a 99.5% (Bonferroni, 10 comparacoes) de NLL(t)-NLL(0) por bloco, artigo 10 (cache/*.npz)."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = lambda k: np.load(os.path.join(ROOT, "cache", f"{k}__ppl40_100.npz"))["x"]
rng = np.random.default_rng(0); out = []
b = C("q_lerp_0.0")
for t in ["0.1","0.2","0.3","0.4","0.5"]:
    d = C(f"q_lerp_{t}") - b
    rng = np.random.default_rng(0); m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    lo, hi = np.percentile(m, [0.25, 99.75]); out.append(f"A10 NLL t={t} - t=0, 99.5% CI: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
print("\n".join(out)); open(os.path.join(ROOT,"results","ci995_art10.txt"),"w").write("\n".join(out)+"\n")
