"""Análise de todos os artigos. Saídas: results/summary.json, results/summary_artNN.txt, figs/figNN_*.pdf"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import key_path, ROOT, mac
sys.path.insert(0, ROOT); from common import stats as S, evalkit as E
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 8, "axes.grid": True, "grid.alpha": .3})
TASKS = E.TASKS; OUT = {}; TXT = {i: [] for i in range(1, 11)}
def ex(k, sp): return os.path.exists(key_path(k, sp))
def ld(k, sp): z = np.load(key_path(k, sp)); return {t: z[t] for t in TASKS}
def say(i, s): TXT[i].append(s); print(f"[{i}] {s}")
def ci(res): # IC bootstrap do macro
    rng = np.random.default_rng(0); ms = [np.mean([r[rng.integers(0, len(r), len(r))].mean() for r in res.values()]) for _ in range(2000)]
    return mac(res), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
def pd(a, b): return S.macro_paired(a, b, B=3000)
def fmt(d): return f"{d[0]*100:+.2f} pp [{d[1]*100:+.2f}, {d[2]*100:+.2f}] p={d[3]:.3f}"
def figsave(name): plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figs", name), bbox_inches="tight"); plt.close()

def curve(prefix, ts, art, label):
    val = {t: mac(ld(f"{prefix}_{t}", "val")) for t in ts}; te = {t: ld(f"{prefix}_{t}", "test") for t in ts}
    t0, t1 = ts[0], ts[-1]; orig = max([t0, t1], key=lambda t: val[t]); ts_best = max(ts, key=lambda t: val[t])
    say(art, f"{label}: val macro por t: " + ", ".join(f"{t}:{val[t]:.4f}" for t in ts))
    for t in ts: m, lo, hi = ci(te[t]); say(art, f"  test t={t}: {m:.4f} [{lo:.4f},{hi:.4f}]")
    say(art, f"  melhor original na validação: t={orig}; t* da validação: t={ts_best}")
    d = pd(te[ts_best], te[orig]); say(art, f"  t* vs melhor original (teste, pareado): {fmt(d)}")
    for t in (t0, t1): say(art, f"  t* vs t={t}: {fmt(pd(te[ts_best], te[t]))}")
    mx = max(ts, key=lambda t: mac(te[t])); say(art, f"  (oráculo no teste, não usável: t={mx} {mac(te[mx]):.4f}; vs melhor original {fmt(pd(te[mx], te[orig]))})")
    return val, te, orig, ts_best, d
# ---- A1
ts = [f"{t:.1f}" for t in np.arange(0, 1.01, .1)]
v1, te1, orig1, tb1, d1 = curve("q_lerp", ts, 1, "Qwen LERP")
OUT["a1"] = dict(t_star=tb1, orig=orig1, diff=d1)
fig, ax = plt.subplots(figsize=(3.5, 2.4)); x = [float(t) for t in ts]
m = [ci(te1[t]) for t in ts]; ax.errorbar(x, [a[0] for a in m], yerr=[[a[0]-a[1] for a in m], [a[2]-a[0] for a in m]], fmt="o-", ms=3, lw=1, capsize=2, label="test (95% CI)")
ax.plot(x, [v1[t] for t in ts], "s--", ms=3, lw=1, label="validation"); ax.set_xlabel("interpolation $t$ (0=base, 1=instruct)"); ax.set_ylabel("macro accuracy"); ax.legend(fontsize=6); figsave("fig01_lerp_qwen.pdf")
# ---- A2
ts2 = ["0.2", "0.4", "0.6", "0.8"]; rows = []
for t in ts2:
    if ex(f"q_slerp_{t}", "test"):
        d = pd(ld(f"q_slerp_{t}", "test"), te1[t]); rows.append((float(t), d)); say(2, f"SLERP-LERP t={t}: teste {mac(ld(f'q_slerp_{t}','test')):.4f} vs {mac(te1[t]):.4f}; {fmt(d)}")
v2 = {t: mac(ld(f"q_slerp_{t}", "val")) for t in ts2}; b2 = max(v2, key=lambda t: v2[t])
say(2, "val SLERP: " + ", ".join(f"{t}:{v2[t]:.4f}" for t in ts2)); say(2, f"SLERP t* (val)={b2}: vs melhor original (base/instruct escolhido na val): {fmt(pd(ld(f'q_slerp_{b2}','test'), te1[orig1]))}")
OUT["a2"] = dict(rows=[(r[0], r[1]) for r in rows])
fig, ax = plt.subplots(figsize=(3.5, 2.4)); ax.errorbar([r[0] for r in rows], [r[1][0]*100 for r in rows], yerr=[[ (r[1][0]-r[1][1])*100 for r in rows], [(r[1][2]-r[1][0])*100 for r in rows]], fmt="o", capsize=3); ax.axhline(0, c="k", lw=.7)
ax.set_xlabel("$t$"); ax.set_ylabel("SLERP $-$ LERP, test (pp)"); figsave("fig02_slerp_vs_lerp.pdf")
# ---- A3
lams = ["0.5", "0.75", "1.25", "1.5", "2.0"]; v3 = {"0.0": v1["0.0"], "1.0": v1["1.0"]}; t3 = {"0.0": te1["0.0"], "1.0": te1["1.0"]}
for l in lams:
    v3[l] = mac(ld(f"q_ta_{l}", "val")); t3[l] = ld(f"q_ta_{l}", "test")
order = sorted(v3, key=float); say(3, "val por lambda: " + ", ".join(f"{l}:{v3[l]:.4f}" for l in order))
for l in order: m_, lo, hi = ci(t3[l]); say(3, f"  test lambda={l}: {m_:.4f} [{lo:.4f},{hi:.4f}]")
lb = max(order, key=lambda l: v3[l]); say(3, f"lambda* (val)={lb}; vs melhor original (t={orig1}): {fmt(pd(t3[lb], te1[orig1]))}")
ext = max(["1.25", "1.5", "2.0"], key=lambda l: v3[l]); say(3, f"melhor extrapolado na val={ext}: vs instruct (lambda=1): {fmt(pd(t3[ext], t3['1.0']))}")
OUT["a3"] = dict(lam_star=lb)
fig, ax = plt.subplots(figsize=(3.5, 2.4)); xs = [float(l) for l in order]; mm = [ci(t3[l]) for l in order]
ax.errorbar(xs, [a[0] for a in mm], yerr=[[a[0]-a[1] for a in mm], [a[2]-a[0] for a in mm]], fmt="o-", ms=3, capsize=2, label="test"); ax.plot(xs, [v3[l] for l in order], "s--", ms=3, label="validation"); ax.set_xlabel("$\\lambda$ in base+$\\lambda$(instruct$-$base)"); ax.set_ylabel("macro accuracy"); ax.legend(fontsize=6); figsave("fig03_task_arith.pdf")
# ---- A4
v4 = {}; t4 = {}
v4["coder"] = mac(ld("q_coder", "val")); t4["coder"] = ld("q_coder", "test")
g = [f"q_3w_{a}_{b}" for a in (.25, .5, .75) for b in (.25, .5, .75)]
for k in g: v4[k] = mac(ld(k, "val"))
gb = max(g, key=lambda k: v4[k]); say(4, "grade 3 modelos (val): " + ", ".join(f"{k[5:]}:{v4[k]:.3f}" for k in g))
for k in [gb, "q_ic_avg", "q_ties_0.5"]:
    if ex(k, "test"): t4[k] = ld(k, "test")
    v4[k] = mac(ld(k, "val"))
dv = [mac(ld(f"q_dare_s{s}", "val")) for s in range(5)]; dt = [mac(ld(f"q_dare_s{s}", "test")) for s in range(5) if ex(f"q_dare_s{s}", "test")]
for k, r in t4.items(): m_, lo, hi = ci(r); say(4, f"  test {k}: {m_:.4f} [{lo:.4f},{hi:.4f}]  (val {v4[k]:.4f}); vs melhor original: {fmt(pd(r, te1[orig1]))}")
say(4, f"DARE 5 sementes: val {np.mean(dv):.4f}±{np.std(dv, ddof=1):.4f}; teste {np.mean(dt):.4f}±{np.std(dt, ddof=1):.4f} (n={len(dt)}); melhor original teste {mac(te1[orig1]):.4f}")
dres = ld("q_dare_s0", "test"); say(4, f"  DARE s0 vs melhor original: {fmt(pd(dres, te1[orig1]))}")
OUT["a4"] = dict(grid_best=gb)
names = ["base", "instruct", "coder", "3way*", "I+C avg", "TIES", "DARE"]; vals = [mac(te1["0.0"]), mac(te1["1.0"]), mac(t4["coder"]), mac(t4[gb]), mac(t4["q_ic_avg"]), mac(t4["q_ties_0.5"]), float(np.mean(dt))]
fig, ax = plt.subplots(figsize=(3.5, 2.4)); ax.bar(names, vals, color=["#888", "#888", "#c55", "#58a", "#58a", "#58a", "#58a"]); ax.set_ylabel("test macro accuracy"); plt.setp(ax.get_xticklabels(), rotation=35, ha="right"); figsave("fig04_threeway.pdf")
# ---- A5
js = json.load(open(os.path.join(ROOT, "results", "a5_search.json"))); fin = js["final"]; key5 = "q_lw_" + "_".join(f"{x:.2f}" for x in fin)
say(5, f"busca por grupos de camadas: início t={js['start_t']}, final {fin}; val {js['val']:.4f} vs t* global val {v1[tb1]:.4f} (ganho val {(js['val']-v1[tb1])*100:+.2f} pp); n avaliações={len(js['log'])}")
if ex(key5, "test"):
    r5 = ld(key5, "test"); say(5, f"  test camadas: {mac(r5):.4f}; vs t* global: {fmt(pd(r5, te1[tb1]))}; vs melhor original: {fmt(pd(r5, te1[orig1]))}")
    OUT["a5"] = dict(gain_val=js['val'] - v1[tb1], gain_test=pd(r5, te1[tb1]))
    fig, ax = plt.subplots(figsize=(3.5, 2.4)); d5 = OUT["a5"]["gain_test"]; ax.bar(["validation", "test"], [OUT["a5"]["gain_val"]*100, d5[0]*100], yerr=[[0, (d5[0]-d5[1])*100], [0, (d5[2]-d5[0])*100]], capsize=3); ax.set_ylabel("gain over global $t^*$ (pp)"); figsave("fig05_layerwise.pdf")
# ---- A6
ts6 = [f"{t:.1f}" for t in np.arange(0, 1.01, .2)]
if all(ex(f"s_lerp_{t}", "test") for t in ts6):
    v6, te6, orig6, tb6, d6 = curve("s_lerp", ts6, 6, "SmolLM2 LERP"); OUT["a6"] = dict(t_star=tb6, orig=orig6, diff=d6)
    fig, ax = plt.subplots(figsize=(3.5, 2.4)); x = [float(t) for t in ts6]; m = [ci(te6[t]) for t in ts6]
    ax.errorbar(x, [a[0] for a in m], yerr=[[a[0]-a[1] for a in m], [a[2]-a[0] for a in m]], fmt="o-", ms=3, capsize=2, label="test"); ax.plot(x, [v6[t] for t in ts6], "s--", ms=3, label="validation"); ax.set_xlabel("$t$"); ax.set_ylabel("macro accuracy"); ax.legend(fontsize=6); figsave("fig06_lerp_smollm2.pdf")
# ---- A7
if all(ex(f"p_avg{k}", "ppl40_100") for k in range(1, 7)):
    pv = {k: np.load(key_path(f"p_avg{k}", "ppl0_40"))["x"] for k in range(1, 7)}; pt = {k: np.load(key_path(f"p_avg{k}", "ppl40_100"))["x"] for k in range(1, 7)}
    kb = min(pv, key=lambda k: pv[k].mean()); say(7, "NLL val: " + ", ".join(f"k={k}:{pv[k].mean():.4f}" for k in pv)); say(7, "NLL test: " + ", ".join(f"k={k}:{pt[k].mean():.4f}" for k in pt))
    for k in range(2, 7): d = S.paired_diff(pt[k], pt[1]); say(7, f"  k={k} - k=1 (NLL, teste, por bloco): {d[0]:+.5f} [{d[1]:+.5f}, {d[2]:+.5f}] p={d[3]:.3f}")
    say(7, f"k* (val)={kb}"); OUT["a7"] = dict(k_star=kb)
    fig, ax = plt.subplots(figsize=(3.5, 2.4)); ks = list(range(1, 7)); ax.errorbar(ks, [pt[k].mean() for k in ks], yerr=[1.96*pt[k].std(ddof=1)/np.sqrt(len(pt[k])) for k in ks], fmt="o-", ms=3, capsize=2, label="test"); ax.plot(ks, [pv[k].mean() for k in ks], "s--", ms=3, label="validation"); ax.set_xlabel("checkpoints averaged $k$"); ax.set_ylabel("WikiText-2 NLL"); ax.legend(fontsize=6); figsave("fig07_pythia_avg.pdf")
# ---- A8/A9: reamostragem da validação sobre a grade do Qwen
rng = np.random.default_rng(0); T = ts; vit = {t: ld(f"q_lerp_{t}", "val") for t in T}
def sub_macro(idx):  # idx: dict tarefa -> índices
    return np.array([np.mean([vit[t][k][idx[k]].mean() for k in TASKS]) for t in T])
tm = np.array([mac(te1[t]) for t in T]); orac = tm.max()
res8 = []; 
for _ in range(1000):
    idx = {k: rng.integers(0, len(vit["0.0"][k]), len(vit["0.0"][k])) for k in TASKS}; sv = sub_macro(idx); p = int(np.argmax(sv))
    res8.append((sv[p] - sv[0], tm[p] - tm[0], sv[p] - tm[p], p))
r8 = np.array(res8); say(8, f"1000 reamostragens da validação (300 itens/tarefa, com reposição): ganho de validação do t escolhido sobre t=0: {r8[:,0].mean()*100:+.2f} pp; ganho real no teste: {r8[:,1].mean()*100:+.2f} pp; viés (val - teste) do t escolhido: {r8[:,2].mean()*100:+.2f} pp [{np.percentile(r8[:,2],2.5)*100:+.2f},{np.percentile(r8[:,2],97.5)*100:+.2f}]")
say(8, "frequência de t escolhido: " + ", ".join(f"{T[j]}:{(r8[:,3]==j).mean():.2f}" for j in range(len(T)) if (r8[:,3]==j).mean() > 0)); OUT["a8"] = dict(bias=float(r8[:,2].mean()), val_gain=float(r8[:,0].mean()), test_gain=float(r8[:,1].mean()))
fig, ax = plt.subplots(figsize=(3.5, 2.4)); ax.hist(r8[:, 2] * 100, bins=30); ax.axvline(0, c="k", lw=.7); ax.set_xlabel("validation score $-$ test score of selected $t$ (pp)"); ax.set_ylabel("resamples"); figsave("fig08_winners_curse.pdf")
ns = [30, 60, 100, 200, 300]; reg = {}; pwin = {}
for n in ns:
    r = []; w = []
    for _ in range(1000):
        idx = {k: rng.choice(len(vit["0.0"][k]), min(n, len(vit["0.0"][k])), replace=False) for k in TASKS}; sv = sub_macro(idx); p = int(np.argmax(sv)); r.append(orac - tm[p]); w.append(tm[p] > max(tm[0], tm[-1]))
    reg[n] = (np.mean(r), np.percentile(r, 2.5), np.percentile(r, 97.5)); pwin[n] = float(np.mean(w)); say(9, f"n_val={n}/tarefa: regret médio {reg[n][0]*100:.2f} pp [{reg[n][1]*100:.2f},{reg[n][2]*100:.2f}]; P(escolhido supera os dois originais no teste)={pwin[n]:.2f}")
OUT["a9"] = dict(reg={n: reg[n][0] for n in ns})
fig, ax = plt.subplots(figsize=(3.5, 2.4)); ax.errorbar(ns, [reg[n][0]*100 for n in ns], yerr=[[ (reg[n][0]-reg[n][1])*100 for n in ns], [(reg[n][2]-reg[n][0])*100 for n in ns]], fmt="o-", capsize=3); ax.set_xlabel("validation items per task"); ax.set_ylabel("selection regret on test (pp)"); figsave("fig09_val_size.pdf")
# ---- A10
if all(ex(f"q_lerp_{t}", "ppl40_100") for t in ts):
    pv = {t: np.load(key_path(f"q_lerp_{t}", "ppl0_40"))["x"].mean() for t in ts}; pt = {t: np.load(key_path(f"q_lerp_{t}", "ppl40_100"))["x"] for t in ts}
    tn = min(pv, key=lambda t: pv[t]); say(10, "NLL val: " + ", ".join(f"{t}:{pv[t]:.4f}" for t in ts)); say(10, "NLL test: " + ", ".join(f"{t}:{pt[t].mean():.4f}" for t in ts))
    say(10, f"t* de NLL (val)={tn}; t* de acurácia (val)={tb1}")
    for t in ("0.5", "1.0"): d = S.paired_diff(pt[t], pt["0.0"]); say(10, f"  NLL t={t} - base (teste, por bloco): {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")
    accs = [mac(te1[t]) for t in ts]; nl = [pt[t].mean() for t in ts]; r = np.corrcoef(accs, nl)[0, 1]; say(10, f"correlação (acurácia teste, NLL teste) ao longo de t: r={r:.3f}")
    OUT["a10"] = dict(t_nll=tn, r=float(r))
    fig, ax = plt.subplots(figsize=(3.5, 2.4)); sc = ax.scatter(nl, accs, c=[float(t) for t in ts], cmap="viridis", s=18); plt.colorbar(sc, label="$t$"); ax.set_xlabel("WikiText-2 NLL (test)"); ax.set_ylabel("macro accuracy (test)"); figsave("fig10_frontier.pdf")
json.dump(OUT, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1, default=float)
for i, l in TXT.items():
    if l: open(os.path.join(ROOT, "results", f"summary_art{i:02d}.txt"), "w").write("\n".join(l) + "\n")
