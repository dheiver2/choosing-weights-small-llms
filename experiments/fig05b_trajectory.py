"""Trajetoria da busca por grupos de camadas (artigo 5): score de validacao por avaliacao. Saida: artigos/figs/fig05b_trajectory.pdf"""
import os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 8, "axes.grid": True, "grid.alpha": .3})
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); js = json.load(open(os.path.join(R, "results", "a5_search.json"))); log = js["log"]
v = [x[1] * 100 for x in log]; best = [max(v[:i + 1]) for i in range(len(v))]
fig, ax = plt.subplots(figsize=(3.5, 2.4)); ax.plot(range(1, len(v) + 1), v, "o", ms=3, label="candidate (validation)"); ax.step(range(1, len(v) + 1), best, where="post", lw=1.2, c="C1", label="best so far")
ax.axhline(v[0], c="k", lw=.7, ls="--", label="global $t^*=0.7$")
for a, b in ((6.5, "g1"), (11.5, "g2"), (16.5, "g3")): ax.axvline(a, c="gray", lw=.5)
ax.set_xlabel("evaluation index (coordinate sweeps: g1 | g2 | g3 | g4)"); ax.set_ylabel("validation macro acc. (%)"); ax.legend(fontsize=6, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(R, "artigos", "figs", "fig05b_trajectory.pdf"), bbox_inches="tight")
