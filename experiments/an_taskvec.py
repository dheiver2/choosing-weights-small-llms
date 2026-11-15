"""Exploratório: normas e cosseno dos vetores de tarefa. Saída results/taskvec.json"""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from engine import *
def stats(a, b, c=None):
    na = nb = nc = dab = dac = dbc = 0.0; ab = 0.0
    sa = sb = sc = dd = 0.0
    for k in a:
        d1 = (b[k] - a[k]).flatten().double(); n0 = a[k].flatten().double().pow(2).sum().item(); sa += n0
        sb += d1.pow(2).sum().item()
        if c is not None:
            d2 = (c[k] - a[k]).flatten().double(); sc += d2.pow(2).sum().item(); dd += (d1 @ d2).item()
    out = {"rel_norm_1": (sb / sa) ** .5}
    if c is not None: out.update(rel_norm_2=(sc / sa) ** .5, cos=dd / ((sb * sc) ** .5 + 1e-12))
    return out
R = {}
b, i, c = W("qwen25-05b"), W("qwen25-05b-instruct"), W("qwen25-coder-05b-instruct")
R["qwen_inst_coder"] = stats(b, i, c); del c
sb_, si_ = W("smollm2-360m"), W("smollm2-360m-instruct"); R["smol_inst"] = stats(sb_, si_)
json.dump(R, open(os.path.join(ROOT, "results", "taskvec.json"), "w"), indent=1); print(R)
