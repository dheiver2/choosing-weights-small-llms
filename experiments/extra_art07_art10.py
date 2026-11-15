"""Estatisticas extras dos artigos 7 e 10 (a partir de cache/*.npz): diferencas consecutivas em k e NLL(t)-NLL(0) por bloco."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "common"))
import stats as S
C = lambda k, sp: np.load(os.path.join(ROOT, "cache", f"{k}__{sp}.npz"))["x"]
out = []
def say(s): out.append(s); print(s)
for k in range(2, 7):
    d = S.paired_diff(C(f"p_avg{k}", "ppl40_100"), C(f"p_avg{k-1}", "ppl40_100")); say(f"A7 consecutive k={k} - k={k-1} (test, per block): {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
for k in (1, 5, 6): x = C(f"p_avg{k}", "ppl40_100"); say(f"A7 k={k} test blocks: mean {x.mean():.4f} sd {x.std(ddof=1):.4f} min {x.min():.4f} max {x.max():.4f}")
d6 = C("p_avg6", "ppl40_100") - C("p_avg1", "ppl40_100"); say(f"A7 k=6 vs k=1: fraction of test blocks with lower NLL = {(d6<0).mean():.2f}")
for t in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]:
    d = S.paired_diff(C(f"q_lerp_{t}", "ppl40_100"), C("q_lerp_0.0", "ppl40_100")); say(f"A10 NLL t={t} - t=0 (test, per block): {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
A = lambda t: {k: np.load(os.path.join(ROOT, "cache", f"q_lerp_{t}__test.npz"))[k] for k in ("arc_easy", "arc_challenge", "hellaswag", "winogrande")}
for t in ("0.7", "1.0"):
    d = S.macro_paired(A(t), A("0.0")); say(f"A10 macro accuracy t={t} - t=0 (test, paired by item): {d[0]*100:+.2f} pp [{d[1]*100:+.2f}, {d[2]*100:+.2f}] p={d[3]:.3f}")
open(os.path.join(ROOT, "results", "extra_art07_art10.txt"), "w").write("\n".join(out) + "\n")
