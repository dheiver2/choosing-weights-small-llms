"""Tabelas dos artigos 4 e 5 (numeros lidos do cache por item; nenhum numero digitado a mao)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import key_path, ROOT, mac
sys.path.insert(0, ROOT); from common import stats as S, evalkit as E
T = E.TASKS
def ld(k, sp): z = np.load(key_path(k, sp)); return {t: z[t] for t in T}
def ex(k, sp): return os.path.exists(key_path(k, sp))
def ci(res):
    rng = np.random.default_rng(0); ms = [np.mean([r[rng.integers(0, len(r), len(r))].mean() for r in res.values()]) for _ in range(2000)]
    return mac(res), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
def pdiff(a, b): return S.macro_paired(a, b, B=3000)
base_t = ld("q_lerp_0.0", "test"); out = {}
def row(name, k):
    v = mac(ld(k, "val")); r = {"name": name, "val": v}
    if ex(k, "test"):
        t = ld(k, "test"); m, lo, hi = ci(t); d = pdiff(t, base_t); r.update(test=m, lo=lo, hi=hi, d=d[0], dlo=d[1], dhi=d[2], p=d[3])
    return r
rows4 = [row("Qwen2.5-0.5B (base)", "q_lerp_0.0"), row("Qwen2.5-0.5B-Instruct", "q_lerp_1.0"), row("Coder-0.5B-Instruct", "q_coder"), row("LERP(base,instruct), t=0.7", "q_lerp_0.7")]
for a in (.25, .5, .75):
    for b in (.25, .5, .75): rows4.append(row(f"3-way $\\lambda_I$={a}, $\\lambda_C$={b}", f"q_3w_{a}_{b}"))
rows4 += [row("Avg(instruct, coder)", "q_ic_avg"), row("TIES (d=0.5)", "q_ties_0.5")]
for s in range(5): rows4.append(row(f"DARE (p=0.5), seed {s}", f"q_dare_s{s}"))
out["art4"] = rows4
b3 = ld("q_3w_0.75_0.75", "test"); out["art4_3w_vs_lerp07"] = pdiff(b3, ld("q_lerp_0.7", "test"))
out["art4_3w_vs_coder"] = pdiff(b3, ld("q_coder", "test"))
dt = [mac(ld(f"q_dare_s{s}", "test")) for s in range(5)]; dv = [mac(ld(f"q_dare_s{s}", "val")) for s in range(5)]
out["dare"] = dict(test_mean=float(np.mean(dt)), test_sd=float(np.std(dt, ddof=1)), val_mean=float(np.mean(dv)), val_sd=float(np.std(dv, ddof=1)), test_min=min(dt), test_max=max(dt))
# per-task test accuracy for key models
out["pertask"] = {k: {t: float(ld(k, "test")[t].mean()) for t in T} for k in ["q_lerp_0.0", "q_lerp_1.0", "q_coder", "q_3w_0.75_0.75", "q_ic_avg", "q_ties_0.5"]}
# art5
js = json.load(open(os.path.join(ROOT, "results", "a5_search.json"))); fin = js["final"]; k5 = "q_lw_" + "_".join(f"{x:.2f}" for x in fin)
t5 = ld(k5, "test"); tg = ld("q_lerp_0.7", "test")
out["a5"] = dict(final=fin, val=js["val"], val_global=mac(ld("q_lerp_0.7", "val")), test=ci(t5), test_global=ci(tg), test_base=ci(base_t), test_inst=ci(ld("q_lerp_1.0", "test")),
    vs_global=pdiff(t5, tg), vs_base=pdiff(t5, base_t), vs_inst=pdiff(t5, ld("q_lerp_1.0", "test")))
out["a5_log"] = [(l, v) for l, v in js["log"]]
out["a5_pertask"] = {n: {t: float(r[t].mean()) for t in T} for n, r in [("layer", t5), ("global", tg), ("base", base_t)]}
# n of items per task
out["n_test"] = {t: int(len(base_t[t])) for t in T}; out["n_val"] = {t: int(len(ld("q_lerp_0.0", "val")[t])) for t in T}
json.dump(out, open(os.path.join(ROOT, "results", "tables_art04_05.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
# extra art5: diferenca pareada NA VALIDACAO (o proprio ganho de selecao tem ruido)
vl = ld(k5, "val"); vg = ld("q_lerp_0.7", "val"); vb = ld("q_lerp_0.0", "val")
ex5 = dict(val_layer_vs_global=pdiff(vl, vg), val_layer_vs_base=pdiff(vl, vb), val_global_vs_base=pdiff(vg, vb), val_base=mac(vb), val_layer=mac(vl), val_global=mac(vg),
           test_global_vs_base=pdiff(tg, base_t))
d = json.load(open(os.path.join(ROOT, "results", "tables_art04_05.json"))); d["a5_extra"] = ex5
json.dump(d, open(os.path.join(ROOT, "results", "tables_art04_05.json"), "w"), indent=1, default=float); print(json.dumps(ex5, indent=1, default=float))
