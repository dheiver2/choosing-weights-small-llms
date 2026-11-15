"""Artigo 8: diferenca entre o ganho de teste do t escolhido pela validacao e o ganho medio de um t uniforme na grade,
com intervalo devido ao ruido dos itens de teste apenas.
Escolhas t_b: 1000 reamostras da validacao (np.random.default_rng(0), mesma ordem de sorteio do analyze.py), fixas.
Ruido de teste: 2000 reamostras bootstrap dos itens de teste dentro da tarefa (np.random.default_rng(4)).
Por reamostra de teste r: D_r = mean_b g_r(t_b) - mean_t g_r(t), g_r(t) = A_r(t) - A_r(0).
Saida: results/summary_art08_selection_vs_random.txt. Uso: python3 experiments/an_art08_selection_vs_random.py"""
import os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]; T = [f"{t:.1f}" for t in np.arange(0, 1.01, .1)]
V = {t: {k: np.load(f"{ROOT}/cache/q_lerp_{t}__val.npz")[k] for k in TASKS} for t in T}
X = {t: {k: np.load(f"{ROOT}/cache/q_lerp_{t}__test.npz")[k] for k in TASKS} for t in T}
mac = lambda D, t, idx: np.mean([D[t][k][idx[k]].mean() for k in TASKS])
rng = np.random.default_rng(0); ch = []
for _ in range(1000):
    idx = {k: rng.integers(0, len(V["0.0"][k]), len(V["0.0"][k])) for k in TASKS}
    ch.append(int(np.argmax([mac(V, t, idx) for t in T])))
freq = np.bincount(ch, minlength=11) / len(ch)
out = []
def say(s): out.append(s); print(s)
fx = {k: np.arange(len(X["0.0"][k])) for k in TASKS}
g = np.array([mac(X, t, fx) for t in T]); g = (g - g[0]) * 100
d0 = freq @ g - g.mean()
say("seed 0 (validation resamples, B=1000, same draw order as analyze.py); test-item bootstrap seed 4, R=2000")
say(f"point estimate on the observed test set: selected {freq@g:+.2f} pp, uniform t {g.mean():+.2f} pp, difference {d0:+.2f} pp")
rt = np.random.default_rng(4); D = []
for _ in range(2000):
    ix = {k: rt.integers(0, len(X["0.0"][k]), len(X["0.0"][k])) for k in TASKS}
    a = np.array([mac(X, t, ix) for t in T]); a = (a - a[0]) * 100
    D.append(freq @ a - a.mean())
D = np.array(D)
say(f"selected minus uniform test gain, test noise only: mean {D.mean():+.2f} pp, 95% percentile interval [{np.percentile(D,2.5):+.2f}, {np.percentile(D,97.5):+.2f}]; SD {D.std():.2f}")
open(f"{ROOT}/results/summary_art08_selection_vs_random.txt", "w").write("\n".join(out) + "\n")
