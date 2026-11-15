"""Executa todas as avaliações dos 10 artigos (cache por item). Uso: python3 experiments/run_all.py [fase ...]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import *
import engine
Q, QI, QC = "qwen25-05b", "qwen25-05b-instruct", "qwen25-coder-05b-instruct"
S, SI = "smollm2-360m", "smollm2-360m-instruct"
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
def best(d): return max(d, key=lambda k: d[k])   # d: chave -> macro val
def phase_a1():
    b, i = W(Q), W(QI); val = {}
    for t in np.round(np.arange(0, 1.01, .1), 1):
        k = f"q_lerp_{t:.1f}"; st = M.lerp(b, i, float(t)); val[k] = mac(ev(k, st, Q, "val")); evppl(k, st, Q, 0, 40)
    for t in np.round(np.arange(0, 1.01, .1), 1):
        k = f"q_lerp_{t:.1f}"; st = M.lerp(b, i, float(t)); ev(k, st, Q, "test"); evppl(k, st, Q, 40, 100)
def phase_a6():
    b, i = W(S), W(SI)
    for split in ("val", "test"):
        for t in np.round(np.arange(0, 1.01, .2), 1):
            k = f"s_lerp_{t:.1f}"; ev(k, M.lerp(b, i, float(t)), S, split)
def sel_test(prefix_val, builder, host, keys):
    val = {k: mac(ev(k, builder(k), host, "val")) for k in keys}; kb = best(val); ev(kb, builder(kb), host, "test"); return kb, val
def phase_a3():
    b, i = W(Q), W(QI); d = M.delta(i, b); lams = [0.5, 0.75, 1.25, 1.5, 2.0]
    bld = lambda k: M.add(b, d, float(k.split("_")[-1]))
    keys = [f"q_ta_{l}" for l in lams]; kb, v = sel_test("q_ta", bld, Q, keys)
    # testa também todos os lambda para a curva (custo aceitável)
    for k in keys: ev(k, bld(k), Q, "test")
def phase_a2():
    b, i = W(Q), W(QI); bld = lambda k: M.slerp(b, i, float(k.split("_")[-1]))
    keys = [f"q_slerp_{t}" for t in (0.2, 0.4, 0.6, 0.8)]
    for k in keys: ev(k, bld(k), Q, "val")
    for k in keys: ev(k, bld(k), Q, "test")     # grade completa no teste: permite comparar LERP x SLERP ponto a ponto
def dare_state(b, di, dc, s):
    x, y = M.dare(di, .5, s), M.dare(dc, .5, 100 + s)
    return {n: b[n] + .5 * (x[n] + y[n]) for n in b}
def phase_a4():
    b, i, c = W(Q), W(QI), W(QC); di, dc = M.delta(i, b), M.delta(c, b)
    ic = M.lerp(i, c, .5); cc = c; ev('q_coder', cc, Q, 'val'); ev('q_coder', cc, Q, 'test')
    del i, c, cc; engine._sd.pop(QI, None); engine._sd.pop(QC, None); import gc; gc.collect()
    keys = []
    for l1 in (.25, .5, .75):
        for l2 in (.25, .5, .75):
            k = f"q_3w_{l1}_{l2}"; keys.append(k); ev(k, {n: b[n] + l1 * di[n] + l2 * dc[n] for n in b}, Q, "val")
    ik = "q_ic_avg"; ev(ik, ic, Q, "val")
    tk = "q_ties_0.5"; tt = M.ties([di, dc], 0.5, 1.0); ev(tk, M.add(b, tt, 1.0), Q, "val")
    dk = []
    for s in range(5):
        k = f"q_dare_s{s}"; dk.append(k); ev(k, dare_state(b, di, dc, s), Q, "val")
    # teste: melhor da grade, ic_avg, ties, e as 5 sementes DARE
    kb = max(keys, key=lambda k: mac(ev(k, None, Q, "val")))
    l1, l2 = map(float, kb.split("_")[2:]); ev(kb, {n: b[n] + l1 * di[n] + l2 * dc[n] for n in b}, Q, "test")
    ev(ik, ic, Q, "test"); ev(tk, M.add(b, tt, 1.0), Q, "test")
    for s in range(5): ev(f"q_dare_s{s}", dare_state(b, di, dc, s), Q, "test")
def phase_a5():
    b, i = W(Q), W(QI)
    vals = {t: mac(ev(f"q_lerp_{t:.1f}", None, Q, "val")) for t in np.round(np.arange(0, 1.01, .1), 1)}
    ts = max(vals, key=lambda t: vals[t]); tcur = [float(ts)] * 4; tother = float(ts)
    def build(tg):
        out = {}
        for n in b:
            l = M.layer_of(n); g = 3 if l is None else min(l // 6, 3); t = tother if l is None else tg[g]
            out[n] = (1 - t) * b[n] + t * i[n]
        return out
    key = lambda tg: "q_lw_" + "_".join(f"{x:.2f}" for x in tg)
    cur = mac(ev(key(tcur), build(tcur), Q, "val")); log = [(list(tcur), cur)]
    for g in range(4):
        for cand in (0, .25, .5, .75, 1.0):
            if abs(cand - tcur[g]) < 1e-9: continue
            tg = list(tcur); tg[g] = cand; sc = mac(ev(key(tg), build(tg), Q, "val")); log.append((tg, sc))
            if sc > cur: cur, bestc = sc, cand; tcur = tg
    json.dump({"start_t": float(ts), "final": tcur, "val": cur, "log": log}, open(os.path.join(RES, "a5_search.json"), "w"))
    ev(key(tcur), build(tcur), Q, "test")
def phase_a7():
    steps = [143000, 140000, 130000, 120000, 110000, 100000]; sds = [W(f"pythia160m-step{s}") for s in steps]; H = "pythia160m-step143000"
    for k in range(1, 7):
        st = M.average(sds[:k]); evppl(f"p_avg{k}", st, H, 0, 40); evppl(f"p_avg{k}", st, H, 40, 100)
PH = {"a1": phase_a1, "a6": phase_a6, "a3": phase_a3, "a2": phase_a2, "a4": phase_a4, "a5": phase_a5, "a7": phase_a7}
if __name__ == "__main__":
    for p in (sys.argv[1:] or list(PH)):
        print("== fase", p, flush=True); PH[p]()
        if p in ("a1", "a6"): engine._sd.clear()
    print("FIM", flush=True)
