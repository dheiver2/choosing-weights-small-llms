"""Calculos complementares dos artigos 8 e 9 (nao alteram results/summary_art08/09.txt).
Reusa a mesma grade LERP do Qwen (cache/q_lerp_{t}__val|test.npz). Saida: results/summary_art0809_extra.txt e figs/fig08b_gain_gap.pdf
Uso: python3 experiments/an_art0809_extra.py"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 8, "axes.grid": True, "grid.alpha": .3})
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]; T = [f"{t:.1f}" for t in np.arange(0, 1.01, .1)]
def ld(t, sp): z = np.load(f"{ROOT}/cache/q_lerp_{t}__{sp}.npz"); return {k: z[k] for k in TASKS}
V = {t: ld(t, "val") for t in T}; X = {t: ld(t, "test") for t in T}
out = []
def say(s): out.append(s); print(s)
def macro(D, t, idx): return np.mean([D[t][k][idx[k]].mean() for k in TASKS])
full = lambda D: {k: np.arange(len(D[T[0]][k])) for k in TASKS}
iv, ix = full(V), full(X)
tm = np.array([macro(X, t, ix) for t in T]); vm = np.array([macro(V, t, iv) for t in T])
say("test macro por t: " + ", ".join(f"{t}:{m:.4f}" for t, m in zip(T, tm)))
say(f"faixa do teste (max-min) na grade de 11 pontos: {(tm.max()-tm.min())*100:.2f} pp; argmax teste t={T[int(tm.argmax())]}, argmin t={T[int(tm.argmin())]}")
say(f"faixa da validacao (max-min): {(vm.max()-vm.min())*100:.2f} pp; argmax val t={T[int(vm.argmax())]}")
orac = tm.max(); reg_t = orac - tm
say(f"regret de escolher t ao acaso (uniforme nos 11 pontos): {reg_t.mean()*100:.2f} pp; sempre t=0: {reg_t[0]*100:.2f} pp; sempre t=1: {reg_t[-1]*100:.2f} pp; mediana por ponto: {np.median(reg_t)*100:.2f} pp; pior ponto: {reg_t.max()*100:.2f} pp")
say(f"fracao dos 11 pontos que supera ambos os originais no teste (t=0 e t=1): {np.mean(tm > max(tm[0], tm[-1])):.2f} ({int((tm > max(tm[0], tm[-1])).sum())}/11)")
# ruido de um unico ponto do teste: bootstrap de itens
rng = np.random.default_rng(1)
bs = np.array([[np.mean([X[t][k][rng.integers(0, len(X[t][k]), len(X[t][k]))].mean() for k in TASKS]) for t in T[:1]] for _ in range(1000)]).ravel()
say(f"erro-padrao bootstrap da acuracia macro de teste (t=0, 1000 reamostras de itens): {bs.std()*100:.2f} pp")
# artigo 8: ganho de validacao vs ganho de teste (mesma semente do analyze.py: rng 0, mesma ordem de sorteio)
rng = np.random.default_rng(0); R = []
for _ in range(1000):
    idx = {k: rng.integers(0, len(V["0.0"][k]), len(V["0.0"][k])) for k in TASKS}
    sv = np.array([macro(V, t, idx) for t in T]); p = int(sv.argmax()); R.append((sv[p]-sv[0], tm[p]-tm[0], p))
R = np.array(R); vg, tg, ch = R[:, 0]*100, R[:, 1]*100, R[:, 2].astype(int)
gap = vg - tg
say(f"[8] ganho val medio {vg.mean():.2f} pp (MC-SE {vg.std()/np.sqrt(1000):.2f}); ganho teste medio {tg.mean():.2f} pp (MC-SE {tg.std()/np.sqrt(1000):.2f}); razao {vg.mean()/tg.mean():.2f}")
say(f"[8] ganho val: mediana {np.median(vg):.2f}, P(ganho val = 0, ou seja t=0 escolhido) = {np.mean(vg==0):.2f}; ganho val min {vg.min():.2f} max {vg.max():.2f}")
say(f"[8] ganho teste do t escolhido: mediana {np.median(tg):.2f}; P(ganho teste<0)={np.mean(tg<0):.2f}; P(ganho teste=0)={np.mean(tg==0):.2f}; P(ganho teste>0)={np.mean(tg>0):.2f}")
say(f"[8] diferenca de ganhos (val - teste) por reamostra: media {gap.mean():.2f} pp; percentis 2.5/97.5 da distribuicao: [{np.percentile(gap,2.5):.2f}, {np.percentile(gap,97.5):.2f}]; P(gap>0)={np.mean(gap>0):.2f}; P(gap<0)={np.mean(gap<0):.2f}")
# bootstrap duplo: reamostra tambem os itens de teste (incerteza do ganho de teste)
rng = np.random.default_rng(2); G = []
for _ in range(1000):
    iv_ = {k: rng.integers(0, len(V["0.0"][k]), len(V["0.0"][k])) for k in TASKS}; ix_ = {k: rng.integers(0, len(X["0.0"][k]), len(X["0.0"][k])) for k in TASKS}
    sv = np.array([macro(V, t, iv_) for t in T]); p = int(sv.argmax())
    G.append(((sv[p]-sv[0])*100, (macro(X, T[p], ix_)-macro(X, "0.0", ix_))*100))
G = np.array(G); gg = G[:, 0] - G[:, 1]
say(f"[8] bootstrap duplo (val e teste reamostrados, 1000x): ganho val medio {G[:,0].mean():.2f}; ganho teste medio {G[:,1].mean():.2f}; diferenca media {gg.mean():.2f} pp, IC95 percentil [{np.percentile(gg,2.5):.2f}, {np.percentile(gg,97.5):.2f}]; P(diferenca>0)={np.mean(gg>0):.3f}")
say(f"[8] bootstrap duplo: IC95 percentil do ganho de teste [{np.percentile(G[:,1],2.5):.2f}, {np.percentile(G[:,1],97.5):.2f}]; do ganho de val [{np.percentile(G[:,0],2.5):.2f}, {np.percentile(G[:,0],97.5):.2f}]")
say(f"[8] P(ganho de teste do t escolhido >= +0.65 pp, o valor do t=0.7 da validacao completa) = {np.mean(tg>=0.65-1e-9):.2f}")
say("[8] contagem de escolhas por t: " + ", ".join(f"{T[j]}:{int((ch==j).sum())}" for j in range(11)))
# Referencia: ganho de teste se escolhesse o t ao acaso entre os 10 diferentes de 0
say(f"[8] ganho de teste medio de um t escolhido ao acaso (11 pontos, incl. t=0): {(tm-tm[0]).mean()*100:.2f} pp; ganho de teste do t=0.7: {(tm[7]-tm[0])*100:.2f} pp")
fig, ax = plt.subplots(figsize=(3.5, 2.4)); b = np.linspace(-1, 4, 41)
ax.hist(vg, bins=b, alpha=.6, label="validation gain"); ax.hist(tg, bins=b, alpha=.6, label="test gain")
ax.axvline(vg.mean(), c="C0", lw=1); ax.axvline(tg.mean(), c="C1", lw=1); ax.set_xlabel("gain of selected $t$ over $t=0$ (pp)"); ax.set_ylabel("resamples"); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig(f"{ROOT}/figs/fig08b_gain_gap.pdf", bbox_inches="tight"); plt.close()
# artigo 9: erro Monte Carlo e distribuicao de escolhas
rng = np.random.default_rng(0); sizes = [30, 60, 100, 200]
say("[9] regret por n (sorteio sem reposicao, 1000x, semente 0 propria deste script): media [percentis 2.5-97.5 da distribuicao] (MC-SE da media); P(regret=0); P(escolher t=0.7 = oraculo)")
rows = []
for n in sizes:
    r = []; c = []
    for _ in range(1000):
        idx = {k: rng.choice(len(V["0.0"][k]), n, replace=False) for k in TASKS}; sv = np.array([macro(V, t, idx) for t in T]); p = int(sv.argmax()); r.append(orac - tm[p]); c.append(p)
    r = np.array(r)*100; c = np.array(c)
    say(f"  n={n}: {r.mean():.2f} [{np.percentile(r,2.5):.2f},{np.percentile(r,97.5):.2f}] (MC-SE {r.std()/np.sqrt(1000):.2f}); P(regret=0)={np.mean(r==0):.2f}; sd={r.std():.2f}; n_t distintos escolhidos={len(set(c))}")
    rows.append(r)
allr = np.concatenate(rows); say(f"  media sobre os 4 n: {allr.mean():.2f} pp; razao regret/regret-aleatorio = {allr.mean()/(reg_t.mean()*100):.2f}")
open(f"{ROOT}/results/summary_art0809_extra.txt", "w").write("\n".join(out) + "\n")

# figura do artigo 9 a partir do summary_art09.txt (n=300 oco: realizacao unica, sem IC) + linha do regret aleatorio
import re
pts = []
for l in open(f"{ROOT}/results/summary_art09.txt"):
    m = re.match(r"n_val=(\d+)/tarefa: regret m.dio ([\d.]+) pp \[([\d.]+),([\d.]+)\]", l)
    if m: pts.append(tuple(float(x) for x in m.groups()))
fig, ax = plt.subplots(figsize=(3.5, 2.4))
a = [p for p in pts if p[0] < 300]; ns_ = [p[0] for p in a]; mu = [p[1] for p in a]
ax.errorbar(ns_, mu, yerr=[[p[1]-p[2] for p in a], [p[3]-p[1] for p in a]], fmt="o-", capsize=3, label="mean (bars: 2.5-97.5% of draws)")
ax.plot([300], [[p for p in pts if p[0] == 300][0][1]], "o", mfc="none", c="C3", label="$n$=300: single realization")
ax.axhline(reg_t.mean()*100, c="k", ls="--", lw=.8, label="random $t$ (11 grid points)")
ax.set_xlabel("validation items per task"); ax.set_ylabel("selection regret on test (pp)"); ax.set_ylim(-0.1, 1.2); ax.legend(fontsize=5.5, loc="upper right")
plt.tight_layout(); plt.savefig(f"{ROOT}/figs/fig09b_val_size.pdf", bbox_inches="tight"); plt.close()
# ganho de teste por t (para a tabela do artigo 8)
with open(f"{ROOT}/results/summary_art0809_extra.txt", "a") as f:
    f.write("[8] ganho de teste por t sobre t=0 (pp): " + ", ".join(f"{t}:{(m-tm[0])*100:+.2f}" for t, m in zip(T, tm)) + "\n")
    f.write("[8] escolhas por t (%): " + ", ".join(f"{T[j]}:{(ch==j).mean()*100:.1f}" for j in range(11)) + "\n")
