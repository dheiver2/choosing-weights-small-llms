"""Artigo 9: regret por n com desempate na validacao (a) np.argmax sobre os floats, exatamente como no analyze.py (reproduz 0.50/0.53/0.51/0.50) e (b) ALEATORIO entre os t com score dentro de 1e-12 do maximo (empate real; o argmax resolve diferencas de 1 ulp de forma arbitraria).
Reproduz a ordem de sorteio do analyze.py: rng = default_rng(0); primeiro 1000 reamostras bootstrap do artigo 8 (consumidas e descartadas),
depois, para n em 30,60,100,200, 1000 subconjuntos sem reposicao (rng.choice por tarefa). Os mesmos subconjuntos servem aos dois desempates.
O desempate aleatorio usa um fluxo separado, default_rng(3), para nao alterar os subconjuntos.
Saida: results/summary_art09_random_tiebreak.txt. Uso: python3 experiments/an_art09_random_tiebreak.py"""
import os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]; T = [f"{t:.1f}" for t in np.arange(0, 1.01, .1)]
V = {t: {k: np.load(f"{ROOT}/cache/q_lerp_{t}__val.npz")[k] for k in TASKS} for t in T}
X = {t: {k: np.load(f"{ROOT}/cache/q_lerp_{t}__test.npz")[k] for k in TASKS} for t in T}
mac = lambda D, t, idx: np.mean([D[t][k][idx[k]].mean() for k in TASKS])
fv = {k: np.arange(len(V["0.0"][k])) for k in TASKS}; fx = {k: np.arange(len(X["0.0"][k])) for k in TASKS}
vfull = np.array([mac(V, t, fv) for t in T]); tm = np.array([mac(X, t, fx) for t in T]); orac = tm.max()
out = []
def say(s): out.append(s); print(s)
tied = [T[j] for j in range(11) if abs(vfull[j] - vfull.max()) < 1e-12]
say(f"full pool (n=300): validation macro max {vfull.max():.4f}; tied at max (t): {tied}")
say(f"test regret by t (pp): " + ", ".join(f"{t}:{(orac-m)*100:.2f}" for t, m in zip(T, tm)))
rng = np.random.default_rng(0); tie = np.random.default_rng(3)
for _ in range(1000): {k: rng.integers(0, len(V["0.0"][k]), len(V["0.0"][k])) for k in TASKS}
say("seed 0 (subsets, same order as analyze.py), seed 3 (random tie-break); 1000 draws per n, without replacement")
for n in [30, 60, 100, 200]:
    ra, rb, ca, cb, nt, nm = [], [], [], [], [], 0
    for _ in range(1000):
        idx = {k: rng.choice(len(V["0.0"][k]), n, replace=False) for k in TASKS}
        sv = np.array([mac(V, t, idx) for t in T]); m = np.flatnonzero(sv >= sv.max() - 1e-12)
        pa = int(sv.argmax()); nm += int(pa != int(m[0])); pb = int(tie.choice(m)); nt.append(len(m))
        ra.append(orac - tm[pa]); rb.append(orac - tm[pb]); ca.append(pa); cb.append(pb)
    ra, rb, ca, cb = np.array(ra)*100, np.array(rb)*100, np.array(ca), np.array(cb)
    f = lambda c, j: int((c == j).sum())
    say(f"n={n}: unrounded mean regret argmax {ra.mean():.3f}, random tie-break {rb.mean():.3f}, difference {ra.mean()-rb.mean():.3f}")
    wn = lambda c: np.mean(tm[c] > max(tm[0], tm[-1]))
    say(f"n={n}: P_win argmax {wn(ca):.2f}, P_win random tie-break {wn(cb):.2f}; argmax != smallest tied t (tolerance 1e-12) in {nm}/1000 draws")
    say(f"n={n}: draws with a tie at the max: {np.mean(np.array(nt)>1)*100:.1f}%")
    say(f"  argmax (analyze.py):   mean regret {ra.mean():.2f} (sd {ra.std():.2f}, MC-SE {ra.std()/np.sqrt(1000):.2f}), P(regret=0)={np.mean(ra==0):.2f}; chose t=0: {f(ca,0)}/1000, t=0.2: {f(ca,2)}/1000, t=0.7: {f(ca,7)}/1000")
    say(f"  random tie-break:   mean regret {rb.mean():.2f} (sd {rb.std():.2f}, MC-SE {rb.std()/np.sqrt(1000):.2f}), P(regret=0)={np.mean(rb==0):.2f}; chose t=0: {f(cb,0)}/1000, t=0.2: {f(cb,2)}/1000, t=0.7: {f(cb,7)}/1000")
say(f"reference: random t regret {(orac-tm).mean()*100:.2f} pp")
open(f"{ROOT}/results/summary_art09_random_tiebreak.txt", "w").write("\n".join(out) + "\n")
