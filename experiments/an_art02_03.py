"""Extras dos artigos 2 e 3. Saida: results/extras_art02_03.json e .txt
 - angulo entre pesos base e instruct por tensor (Qwen2.5-0.5B) e distancia relativa SLERP vs LERP nos pesos
 - diferencas pareadas SLERP-LERP agregadas; contrastes extras do artigo 3 (exploratorio)
Uso: python3 experiments/an_art02_03.py"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import key_path, ROOT, mac, W
sys.path.insert(0, ROOT); from common import stats as S, evalkit as E, mergeops as M
TASKS = E.TASKS; out = {}; txt = []
def say(s): txt.append(s); print(s)
def ld(k, sp): z = np.load(key_path(k, sp)); return {t: z[t] for t in TASKS}
def pd(a, b): return S.macro_paired(a, b, B=3000)
def fmt(d): return f"{d[0]*100:+.2f} pp [{d[1]*100:+.2f}, {d[2]*100:+.2f}] p={d[3]:.3f}"
# --- geometria dos pesos
a, b = W("qwen25-05b"), W("qwen25-05b-instruct")
ang = []; relgap = {t: [0.0, 0.0] for t in (0.2, 0.4, 0.6, 0.8)}; sq_lerp = {t: 0.0 for t in relgap}; sq_gap = {t: 0.0 for t in relgap}
nel = []; maxgap = {t: 0.0 for t in relgap}
for k in a:
    x, y = a[k].flatten().double(), b[k].flatten().double(); nx, ny = x.norm(), y.norm()
    om = float(torch.acos(torch.clamp((x / nx) @ (y / ny), -1, 1))); ang.append(om); nel.append(x.numel())
    for t in relgap:
        lerp = (1 - t) * x + t * y
        if om < 1e-4: sl = lerp
        else: so = np.sin(om); sl = (np.sin((1 - t) * om) / so) * x + (np.sin(t * om) / so) * y
        sq_lerp[t] += float(lerp.pow(2).sum()); sq_gap[t] += float((sl - lerp).pow(2).sum())
        maxgap[t] = max(maxgap[t], float((sl - lerp).norm() / lerp.norm()))
ang = np.array(ang); nel = np.array(nel)
out["angle_rad"] = dict(n_tensors=len(ang), median=float(np.median(ang)), mean=float(ang.mean()), p95=float(np.percentile(ang, 95)), max=float(ang.max()),
                        elem_weighted_mean=float((ang * nel).sum() / nel.sum()), median_deg=float(np.degrees(np.median(ang))), max_deg=float(np.degrees(ang.max())))
out["slerp_lerp_weight_gap"] = {str(t): dict(global_rel=float((sq_gap[t] / sq_lerp[t]) ** .5), max_tensor_rel=maxgap[t]) for t in relgap}
say("angle (rad): " + json.dumps(out["angle_rad"])); say("SLERP vs LERP weight gap: " + json.dumps(out["slerp_lerp_weight_gap"]))
# --- artigo 2: diferencas pareadas agregadas
ts = ["0.2", "0.4", "0.6", "0.8"]; diffs = []
for t in ts:
    s, l = ld(f"q_slerp_{t}", "test"), ld(f"q_lerp_{t}", "test")
    d = pd(s, l); diffs.append(d); say(f"SLERP-LERP test t={t}: {fmt(d)}  SLERP={mac(s):.4f} LERP={mac(l):.4f}")
    sv, lv = ld(f"q_slerp_{t}", "val"), ld(f"q_lerp_{t}", "val"); say(f"   val SLERP={mac(sv):.4f} LERP={mac(lv):.4f}")
    out.setdefault("a2", {})[t] = dict(test_slerp=mac(s), test_lerp=mac(l), val_slerp=mac(sv), val_lerp=mac(lv), diff=d)
    # per-task test accuracy
    out["a2"][t]["task_slerp"] = {k: float(v.mean()) for k, v in s.items()}; out["a2"][t]["task_lerp"] = {k: float(v.mean()) for k, v in l.items()}
# media das 4 diferencas (concatena item-diferencas por t, bootstrap conjunto)
rng = np.random.default_rng(0); dl = []
for t in ts:
    s, l = ld(f"q_slerp_{t}", "test"), ld(f"q_lerp_{t}", "test"); dl.append([np.asarray(s[k], float) - np.asarray(l[k], float) for k in TASKS])
ms = []
for _ in range(3000):
    idx = [[rng.integers(0, len(x), len(x)) for x in dl[0]]][0]  # mesmos indices para todos os t (pareado por item)
    ms.append(np.mean([np.mean([x[i].mean() for x, i in zip(d, idx)]) for d in dl]))
ms = np.array(ms); m4 = float(np.mean([np.mean([x.mean() for x in d]) for d in dl]))
agg = (m4, float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)), float(min(2 * min((ms <= 0).mean(), (ms >= 0).mean()), 1)))
out["a2_mean_over_t"] = agg; say("mean SLERP-LERP over 4 t (test): " + fmt(agg))
# baselines originais
te0, te1_ = ld("q_lerp_0.0", "test"), ld("q_lerp_1.0", "test")
out["orig"] = dict(base_test=mac(te0), inst_test=mac(te1_), base_val=mac(ld("q_lerp_0.0", "val")), inst_val=mac(ld("q_lerp_1.0", "val")))
say("orig: " + json.dumps(out["orig"]))
sb = ld("q_slerp_0.4", "test"); say("t*=0.4 SLERP vs base: " + fmt(pd(sb, te0)) + "; vs instruct: " + fmt(pd(sb, te1_)) + "; LERP0.4 vs base: " + fmt(pd(ld("q_lerp_0.4", "test"), te0)))
# --- artigo 3
t3 = {"0.0": te0, "1.0": te1_}; v3 = {"0.0": mac(ld("q_lerp_0.0", "val")), "1.0": mac(ld("q_lerp_1.0", "val"))}
for l in ["0.5", "0.75", "1.25", "1.5", "2.0"]: t3[l] = ld(f"q_ta_{l}", "test"); v3[l] = mac(ld(f"q_ta_{l}", "val"))
out["a3"] = {}
for l in sorted(t3, key=float):
    d = pd(t3[l], te1_); dv = pd(ld("q_lerp_1.0", "val") if l == "1.0" else (ld("q_lerp_0.0", "val") if l == "0.0" else ld(f"q_ta_{l}", "val")), ld("q_lerp_1.0", "val"))
    out["a3"][l] = dict(val=v3[l], test=mac(t3[l]), test_vs_inst=d, val_vs_inst=dv, task={k: float(v.mean()) for k, v in t3[l].items()})
    say(f"lambda={l}: val={v3[l]:.4f} test={mac(t3[l]):.4f} test-inst {fmt(d)} | val-inst {fmt(dv)}")
say("0.75 vs base (explor.): " + fmt(pd(t3["0.75"], te0)))
for l in ["1.25", "1.5", "2.0"]: say(f"lambda={l} vs lambda=1 (test): {fmt(pd(t3[l], te1_))}; vs base: {fmt(pd(t3[l], te0))}")
json.dump(out, open(os.path.join(ROOT, "results", "extras_art02_03.json"), "w"), indent=1); open(os.path.join(ROOT, "results", "extras_art02_03.txt"), "w").write("\n".join(txt))
