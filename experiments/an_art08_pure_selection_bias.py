"""Artigo 8: vies de selecao puro no mundo bootstrap, V_b(t^) - V(t^).
V = acuracia macro da validacao completa (cache/q_lerp_{t}__val.npz); V_b = reamostra bootstrap.
Mesma semente e mesma ordem de sorteio do analyze.py (np.random.default_rng(0), 1000 reamostras, rng.integers por tarefa).
Saida: results/summary_art08_pure_bias.txt. Uso: python3 experiments/an_art08_pure_selection_bias.py"""
import os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]; T = [f"{t:.1f}" for t in np.arange(0, 1.01, .1)]
V = {t: {k: np.load(f"{ROOT}/cache/q_lerp_{t}__val.npz")[k] for k in TASKS} for t in T}
X = {t: {k: np.load(f"{ROOT}/cache/q_lerp_{t}__test.npz")[k] for k in TASKS} for t in T}
mac = lambda D, t, idx: np.mean([D[t][k][idx[k]].mean() for k in TASKS])
full = {k: np.arange(len(V["0.0"][k])) for k in TASKS}
vfull = np.array([mac(V, t, full) for t in T]); tm = np.array([mac(X, t, {k: np.arange(len(X["0.0"][k])) for k in TASKS}) for t in T])
out = []
def say(s): out.append(s); print(s)
rng = np.random.default_rng(0); R = []
for _ in range(1000):
    idx = {k: rng.integers(0, len(V["0.0"][k]), len(V["0.0"][k])) for k in TASKS}
    sv = np.array([mac(V, t, idx) for t in T]); p = int(sv.argmax())
    R.append((sv[p] - vfull[p], (sv[p] - sv[0]) - (vfull[p] - vfull[0]), sv[p] - sv[0], tm[p] - tm[0], vfull[p] - vfull[0]))
R = np.array(R) * 100
def line(name, x): say(f"{name}: mean {x.mean():+.2f} pp (MC-SE {x.std()/np.sqrt(len(x)):.2f}); 2.5-97.5% of resamples [{np.percentile(x,2.5):+.2f}, {np.percentile(x,97.5):+.2f}]; P(>0)={np.mean(x>0):.2f}")
say("seed 0, B=1000 within-task bootstrap of validation, same draw order as analyze.py")
say(f"full-validation macro by t: " + ", ".join(f"{t}:{v:.4f}" for t, v in zip(T, vfull)))
line("pure selection bias, level V_b(t_hat) - V(t_hat)", R[:, 0])
line("pure selection bias, gain (V_b(t_hat)-V_b(0)) - (V(t_hat)-V(0))", R[:, 1])
line("validation-gain of selected t on the resample, V_b(t_hat)-V_b(0)", R[:, 2])
line("true full-validation gain of selected t, V(t_hat)-V(0)", R[:, 4])
line("fixed-test gain of selected t, A(t_hat)-A(0)", R[:, 3])
say(f"full-validation gain of t=0.7: {(vfull[7]-vfull[0])*100:+.2f} pp; test gain of t=0.7: {(tm[7]-tm[0])*100:+.2f} pp")
say(f"shift term: mean[V(t_hat)-V(0)] - mean[A(t_hat)-A(0)] = {(R[:,4].mean()-R[:,3].mean()):+.2f} pp")
open(f"{ROOT}/results/summary_art08_pure_bias.txt", "w").write("\n".join(out) + "\n")
