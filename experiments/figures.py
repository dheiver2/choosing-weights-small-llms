"""Regenera TODAS as figuras dos artigos a partir de cache/*.npz (mesmos nomes de arquivo usados nos .tex).
Saída: figs/ e artigos/figs/. Uso: python3 experiments/figures.py"""
import os, sys, re, json, shutil, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
from figstyle import *; from common import stats as S
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]; B = 2000
def ld(k, sp): z = np.load(f"{ROOT}/cache/{k}__{sp}.npz"); return {t: z[t] for t in TASKS}
mac = lambda r: float(np.mean([v.mean() for v in r.values()]))
def cim(r, seed=0):
    rng = np.random.default_rng(seed); ms = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in r.values()]) for _ in range(B)]
    return mac(r) * 100, np.percentile(ms, 2.5) * 100, np.percentile(ms, 97.5) * 100
def pdiff(a, b): d = S.macro_paired(a, b, B=3000); return d[0] * 100, d[1] * 100, d[2] * 100, d[3]
def out(fig, name):
    os.makedirs(f"{ROOT}/figs", exist_ok=True); os.makedirs(f"{ROOT}/artigos/figs", exist_ok=True)
    fig.savefig(f"{ROOT}/figs/{name}"); fig.savefig(f"{ROOT}/artigos/figs/{name}"); plt.close(fig); print("ok", name)
def band(ax, x, c, color, label, ls="-", marker="o"):
    m = np.array([a[0] for a in c]); lo = np.array([a[1] for a in c]); hi = np.array([a[2] for a in c])
    ax.fill_between(x, lo, hi, color=color, alpha=.15, lw=0); ax.plot(x, m, ls, color=color, marker=marker, label=label)
def curve2(prefix, ts, fname, xlabel, base_idx_fn=None):
    x = [float(t) for t in ts]; V = {t: ld(f"{prefix}_{t}", "val") for t in ts}; X = {t: ld(f"{prefix}_{t}", "test") for t in ts}
    vm = {t: mac(V[t]) for t in ts}; orig = max([ts[0], ts[-1]], key=lambda t: vm[t]); star = max(ts, key=lambda t: vm[t])
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 3.5), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1], hspace=.12))
    cv = [cim(V[t], 1) for t in ts]; cx = [cim(X[t], 2) for t in ts]
    band(a, x, cx, BLUE, "test (n=3,967)"); band(a, x, cv, ORANGE, "validation (n=1,199)", "--", "s")
    a.axhline(cx[ts.index(orig)][0], color=GRAY, ls=":", lw=.9, label=f"best original ($t$={orig}), test level")
    a.scatter([float(star)], [cx[ts.index(star)][0]], s=46, facecolors="none", edgecolors=RED, linewidths=1.2, zorder=5, label=f"$t^*$={star} (chosen on validation)")
    a.set_ylabel("macro accuracy (%)"); a.legend(loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=2, columnspacing=1.0, handlelength=1.6); panel(a, "(a)")
    d = [pdiff(X[t], X[orig]) if t != orig else (0, 0, 0, 1) for t in ts]
    b.errorbar(x, [e[0] for e in d], yerr=[[e[0] - e[1] for e in d], [e[2] - e[0] for e in d]], fmt="o", color=BLUE, capsize=2, elinewidth=.9)
    i = ts.index(star); b.errorbar([x[i]], [d[i][0]], yerr=[[d[i][0] - d[i][1]], [d[i][2] - d[i][0]]], fmt="o", color=RED, capsize=2.5, elinewidth=1.3, ms=5.5, zorder=5)
    zero(b); b.set_ylabel(f"test gain over $t$={orig}\n(pp, paired 95% CI)"); b.set_xlabel(xlabel); panel(b, "(b)"); out(fig, fname)
curve2("q_lerp", [f"{t:.1f}" for t in np.arange(0, 1.01, .1)], "fig01_lerp_qwen.pdf", "interpolation coefficient $t$ (0 = base, 1 = instruct)")
curve2("s_lerp", [f"{t:.1f}" for t in np.arange(0, 1.01, .2)], "fig06_lerp_smollm2.pdf", "interpolation coefficient $t$ (0 = base, 1 = instruct)")
# ---- fig02 SLERP - LERP
ts2 = ["0.2", "0.4", "0.6", "0.8"]; rows = [pdiff(ld(f"q_slerp_{t}", "test"), ld(f"q_lerp_{t}", "test")) for t in ts2]
fig, ax = plt.subplots(figsize=(W, 2.2)); x = [float(t) for t in ts2]
ax.errorbar(x, [r[0] for r in rows], yerr=[[r[0] - r[1] for r in rows], [r[2] - r[0] for r in rows]], fmt="o", color=BLUE, capsize=3, elinewidth=1)
mu = np.mean([r[0] for r in rows]); ax.axhline(mu, color=ORANGE, ls="--", lw=1); ax.text(.805, mu + .03, f"mean of the four: {mu:+.2f} pp", ha="right", va="bottom", fontsize=6.2, color=ORANGE)
zero(ax); ax.set_xlim(.12, .88); ax.set_xticks(x); ax.set_xlabel("interpolation coefficient $t$"); ax.set_ylabel("SLERP $-$ LERP, test\n(pp, paired 95% CI)"); out(fig, "fig02_slerp_vs_lerp.pdf")
# ---- fig03 extrapolação
lams = ["0.0", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0"]; key = lambda l: "q_lerp_0.0" if l == "0.0" else ("q_lerp_1.0" if l == "1.0" else f"q_ta_{l}")
V = {l: ld(key(l), "val") for l in lams}; X = {l: ld(key(l), "test") for l in lams}; x = [float(l) for l in lams]
fig, (a, b) = plt.subplots(2, 1, figsize=(W, 3.5), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1], hspace=.12))
band(a, x, [cim(X[l], 2) for l in lams], BLUE, "test"); band(a, x, [cim(V[l], 1) for l in lams], ORANGE, "validation", "--", "s")
for ax_ in (a, b): ax_.axvspan(1, 2.1, color="#f2f2f2", zorder=0); ax_.axvline(1, color=GRAY, lw=.8, ls=":")
a.text(1.55, .965, "extrapolation ($\\lambda>1$)", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=6.2, color=GRAY)
a.scatter([0.0], [cim(X["0.0"], 2)[0]], s=46, facecolors="none", edgecolors=RED, linewidths=1.2, zorder=5, label="chosen on validation ($\\lambda$=0)"); a.set_ylabel("macro accuracy (%)"); a.legend(loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=3, columnspacing=1.0, handlelength=1.6); panel(a, "(a)")
d = [pdiff(X[l], X["1.0"]) if l != "1.0" else (0, 0, 0, 1) for l in lams]
b.errorbar(x, [e[0] for e in d], yerr=[[e[0] - e[1] for e in d], [e[2] - e[0] for e in d]], fmt="o", color=BLUE, capsize=2, elinewidth=.9); zero(b)
b.set_ylabel("test gain over instruct\n($\\lambda$=1; pp, paired 95% CI)"); b.set_xlabel("$\\lambda$ in base + $\\lambda$ (instruct $-$ base)"); panel(b, "(b)"); out(fig, "fig03_task_arith.pdf")
# ---- fig04 três modelos (barras horizontais com IC)
g = [f"q_3w_{a}_{b}" for a in (.25, .5, .75) for b in (.25, .5, .75)]; gb = max(g, key=lambda k: mac(ld(k, "val")))
items = [("base", "q_lerp_0.0", GRAY), ("instruct", "q_lerp_1.0", GRAY), ("Coder-instruct", "q_coder", RED), ("weighted sum (val-selected)", gb, BLUE), ("instruct+Coder average", "q_ic_avg", BLUE), ("TIES", "q_ties_0.5", BLUE)]
fig, ax = plt.subplots(figsize=(W, 2.5)); ys = list(range(len(items) + 1))[::-1]
for y, (n, k, c) in zip(ys, items):
    m, lo, hi = cim(ld(k, "test"), 3); ax.barh(y, m, color=c, height=.62, alpha=.9); ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="none", ecolor="#222", capsize=2, elinewidth=.9); ax.text(hi + .6, y, f"{m:.1f}", va="center", fontsize=6.4)
ds = [mac(ld(f"q_dare_s{s}", "test")) * 100 for s in range(5)]; y = ys[-1]; ax.barh(y, np.mean(ds), color=BLUE, height=.62, alpha=.9); ax.scatter(ds, [y] * 5, s=9, color="#222", zorder=4); ax.text(max(ds) + 1.2, y, f"{np.mean(ds):.1f}", va="center", fontsize=6.4)
ax.set_yticks(ys); ax.set_yticklabels([n for n, _, _ in items] + ["DARE (5 seeds, dots)"]); b0 = mac(ld("q_lerp_0.0", "test")) * 100; ax.axvline(b0, color=GRAY, ls=":", lw=.9)
ax.set_xlabel("test macro accuracy (%; bars: 95% CI)"); ax.set_xlim(0, 60); ax.grid(axis="y", visible=False); out(fig, "fig04_threeway.pdf")
# ---- fig05 floresta de ganhos + trajetória
js = json.load(open(f"{ROOT}/results/a5_search.json")); fin = js["final"]; k5 = "q_lw_" + "_".join(f"{x:.2f}" for x in fin)
rows = [("validation: layer-wise $-$ global $t^*$", pdiff(ld(k5, "val"), ld("q_lerp_0.7", "val")), BLUE), ("test: layer-wise $-$ global $t^*$", pdiff(ld(k5, "test"), ld("q_lerp_0.7", "test")), BLUE),
        ("test: layer-wise $-$ best original (base)", pdiff(ld(k5, "test"), ld("q_lerp_0.0", "test")), RED)]
fig, ax = plt.subplots(figsize=(W, 1.9)); ys = [2, 1, 0]
for y, (n, d, c) in zip(ys, rows): ax.errorbar([d[0]], [y], xerr=[[d[0] - d[1]], [d[2] - d[0]]], fmt="o", color=c, capsize=3, elinewidth=1.1); ax.text(d[2] + .12, y, f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]", va="center", fontsize=6.2)
ax.axvline(0, color="#222", lw=.8); ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(-1.4, 4.4); ax.set_ylim(-.6, 2.6); ax.grid(axis="y", visible=False); ax.set_xlabel("gain in macro accuracy (pp, paired 95% CI)")
ax.text(.995, .02, "p=0.017 nominal; Bonferroni limit 0.0083", transform=ax.transAxes, ha="right", fontsize=5.8, color=RED); out(fig, "fig05_layerwise.pdf")
log = js["log"]; v = [x[1] * 100 for x in log]; best = [max(v[:i + 1]) for i in range(len(v))]; n = len(v)
fig, ax = plt.subplots(figsize=(W, 2.3)); ax.plot(range(1, n + 1), v, "o", color=BLUE, ms=3.2, label="candidate (validation)"); ax.step(range(1, n + 1), best, where="post", color=ORANGE, lw=1.4, label="best so far")
ax.axhline(v[0], color=GRAY, ls="--", lw=.9); ax.text(n, v[0] - .06, "global $t^*$=0.7", ha="right", va="top", fontsize=6.2, color=GRAY)
for xb, lab in ((1.5, "group 1\n(layers 0–5)"), (6.5, "group 2\n(6–11)"), (11.5, "group 3\n(12–17)"), (16.5, "group 4\n(18–23)")):
    if xb > 1.5: ax.axvline(xb - 0.0, color="#bbb", lw=.6)
ax.set_xlabel("evaluation index (sweeps over groups 1 | 2 | 3 | 4)"); ax.set_ylabel("validation macro acc. (%)"); ax.legend(loc="lower right"); out(fig, "fig05b_trajectory.pdf")
# ---- fig07 Pythia (fp32): checkpoints isolados vs médias
steps = [143000, 140000, 130000, 120000, 110000, 100000]; Lx = lambda k: np.load(f"{ROOT}/cache/{k}.npz")["x"]
sg = {st: Lx(f"fp32_p_single{st}")[40:] for st in steps}; av = {k: Lx(f"fp32_p_avg{k}")[40:] for k in range(1, 7)}; se = lambda a: 1.96 * a.std(ddof=1) / np.sqrt(len(a))
xs_ = [st / 1000 for st in steps]; fig, (a, b) = plt.subplots(2, 1, figsize=(W, 3.5), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1], hspace=.12))
a.errorbar(xs_, [sg[st].mean() for st in steps], yerr=[se(sg[st]) for st in steps], fmt="s", color=GRAY, capsize=2, label="single checkpoint")
a.errorbar(xs_[1:], [av[k].mean() for k in range(2, 7)], yerr=[se(av[k]) for k in range(2, 7)], fmt="o-", color=BLUE, capsize=2, label="average (143k down to x)")
a.set_ylabel("WikiText-2 NLL (nats, fp32)"); a.set_ylim(3.47, 3.76); a.legend(loc="upper center", ncol=2, fontsize=6.2, columnspacing=1.0); panel(a, "(a)")
dS = [S.paired_diff(sg[st], sg[143000]) for st in steps[1:]]; dA = [S.paired_diff(av[k], av[1]) for k in range(2, 7)]
b.errorbar([x + 1.2 for x in xs_[1:]], [d[0] for d in dS], yerr=[[d[0] - d[1] for d in dS], [d[2] - d[0] for d in dS]], fmt="s", color=GRAY, capsize=2, label="single")
b.errorbar([x - 1.2 for x in xs_[1:]], [d[0] for d in dA], yerr=[[d[0] - d[1] for d in dA], [d[2] - d[0] for d in dA]], fmt="o", color=BLUE, capsize=2, label="average"); zero(b)
b.set_ylabel("NLL change vs. last\ncheckpoint (paired 95% CI)"); b.set_xlabel("oldest checkpoint included (thousand steps)"); b.invert_xaxis(); b.legend(loc="lower left", fontsize=6); panel(b, "(b)"); out(fig, "fig07_pythia_avg.pdf")
# ---- fig08 / fig08b (reamostragem da validação)
T = [f"{t:.1f}" for t in np.arange(0, 1.01, .1)]; V = {t: ld(f"q_lerp_{t}", "val") for t in T}; X = {t: ld(f"q_lerp_{t}", "test") for t in T}
mm = lambda D, t, idx: np.mean([D[t][k][idx[k]].mean() for k in TASKS]); full = lambda D: {k: np.arange(len(D[T[0]][k])) for k in TASKS}
tm = np.array([mm(X, t, full(X)) for t in T]); rng = np.random.default_rng(0); R = []
for _ in range(1000):
    idx = {k: rng.integers(0, len(V["0.0"][k]), len(V["0.0"][k])) for k in TASKS}; sv = np.array([mm(V, t, idx) for t in T]); p = int(sv.argmax()); R.append((sv[p] - sv[0], tm[p] - tm[0], sv[p] - tm[p]))
R = np.array(R); vg, tg, lev = R[:, 0] * 100, R[:, 1] * 100, R[:, 2] * 100
fig, ax = plt.subplots(figsize=(W, 2.3)); bins = np.linspace(-1, 4, 41)
ax.hist(vg, bins=bins, color=ORANGE, alpha=.55, label=f"validation gain (mean {vg.mean():+.2f})"); ax.hist(tg, bins=bins, color=BLUE, alpha=.55, label=f"test gain (mean {tg.mean():+.2f})")
ax.axvline(vg.mean(), color=ORANGE, lw=1.4); ax.axvline(tg.mean(), color=BLUE, lw=1.4); ax.set_xlabel("gain of the selected $t$ over $t=0$ (pp)"); ax.set_ylabel("resamples (of 1000)"); ax.legend(loc="upper right"); out(fig, "fig08b_gain_gap.pdf")
fig, ax = plt.subplots(figsize=(W, 2.1)); ax.hist(lev, bins=30, color=GRAY, alpha=.75); ax.axvline(lev.mean(), color=RED, lw=1.3); ax.axvline(0, color="#222", lw=.8)
ax.set_xlabel("validation score $-$ test score of the selected $t$ (pp; level difference)"); ax.set_ylabel("resamples"); out(fig, "fig08_winners_curse.pdf")
# ---- fig09b
pts = []
for l in open(f"{ROOT}/results/summary_art09.txt"):
    m = re.match(r"n_val=(\d+)/tarefa: regret m.dio ([\d.]+) pp \[([\d.]+),([\d.]+)\]", l)
    if m: pts.append(tuple(float(x) for x in m.group(1, 2, 3, 4)))
reg_t = (tm.max() - tm).mean() * 100; a_ = [p for p in pts if p[0] < 300]
fig, ax = plt.subplots(figsize=(W, 2.4)); ax.errorbar([p[0] for p in a_], [p[1] for p in a_], yerr=[[p[1] - p[2] for p in a_], [p[3] - p[1] for p in a_]], fmt="o-", color=BLUE, capsize=3, label="mean regret (bars: 2.5–97.5% of draws)")
ax.plot([300], [[p for p in pts if p[0] == 300][0][1]], "o", mfc="none", mec=RED, mew=1.2, ms=6, label="$n$=300: single deterministic outcome")
ax.axhline(reg_t, color=GRAY, ls="--", lw=1); ax.text(312, reg_t - .03, f"random choice of $t$ ({reg_t:.2f} pp)", fontsize=6.2, color=GRAY, va="top", ha="right")
ax.set_xlabel("validation items per task"); ax.set_ylabel("selection regret on test (pp)"); ax.set_ylim(-.1, 1.3); ax.set_xlim(15, 315); ax.legend(loc="upper right", fontsize=5.9); out(fig, "fig09b_val_size.pdf")
# ---- fig10 fronteira
import matplotlib.cm as cm
pn = {t: np.load(f"{ROOT}/cache/q_lerp_{t}__ppl40_100.npz")["x"] for t in T}; pv_ = {t: np.load(f"{ROOT}/cache/q_lerp_{t}__ppl0_40.npz")["x"].mean() for t in T}
acc = {t: cim(X[t], 2) for t in T}; tnll = min(T, key=lambda t: pv_[t]); tacc = max(T, key=lambda t: mac(V[t]))
fig, ax = plt.subplots(figsize=(W, 2.7)); xs = [pn[t].mean() for t in T]; ys = [acc[t][0] for t in T]
ax.plot(xs, ys, "-", color="#c4c4c4", lw=1, zorder=1)
sc = ax.scatter(xs, ys, c=[float(t) for t in T], cmap="viridis", s=26, zorder=3, edgecolors="white", linewidths=.5)
for t in (tnll, tacc): i = T.index(t); ax.scatter([xs[i]], [ys[i]], s=70, facecolors="none", edgecolors=RED, linewidths=1.3, zorder=4)
off = {"0.0": (-6, 5), "0.5": (6, -9), "0.6": (-9, 4), "0.7": (4, 6), "1.0": (4, 6), "0.1": (-9, -9)}
for t, (dx, dy) in off.items(): i = T.index(t); ax.annotate(f"$t$={t}", (xs[i], ys[i]), textcoords="offset points", xytext=(dx, dy), fontsize=6, ha="center")
cb = plt.colorbar(sc, ax=ax, pad=.02, fraction=.045); cb.set_label("$t$", fontsize=7); cb.ax.tick_params(labelsize=6)
hw = 1.96 * np.mean([acc[t][2] - acc[t][1] for t in T]) / 2 / 1.96; xl = ax.get_xlim(); yl = ax.get_ylim()
ax.set_ylim(yl[0] - .2, yl[1]); yl = ax.get_ylim(); bx = xl[0] + .90 * (xl[1] - xl[0]); by = yl[0] + .50 * (yl[1] - yl[0])
ax.errorbar([bx], [by], yerr=[[hw]], fmt="none", ecolor="#444", capsize=3, elinewidth=1); ax.text(bx - .02 * (xl[1] - xl[0]), by, "typical 95% CI\nof one point", fontsize=5.8, va="center", ha="right", color="#444")
ax.text(.98, .03, "red rings: best $t$ on validation\nfor NLL ($t$=%s) and accuracy ($t$=%s)" % (tnll, tacc), transform=ax.transAxes, fontsize=5.9, color=RED, ha="right")
ax.set_xlabel("WikiText-2 NLL on test (nats; lower is better)"); ax.set_ylabel("macro accuracy on test (%)"); out(fig, "fig10_frontier.pdf")
