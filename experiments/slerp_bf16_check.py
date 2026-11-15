"""Artigo 2: SLERP e LERP diferem depois do cast para bf16? (a) fração de elementos idênticos após o cast; (b) acurácia de validação em fp32 de SLERP e LERP (t=0.4 e 0.6).
Saída: results/slerp_bf16_check.txt"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import ROOT, mp, W, M, E, mac
from transformers import AutoModelForCausalLM, AutoTokenizer
out = []; say = lambda s: (out.append(s), print(s, flush=True))
b, i = W("qwen25-05b"), W("qwen25-05b-instruct")
for t in (0.4, 0.6):
    L, S = M.lerp(b, i, t), M.slerp(b, i, t); same = tot = 0
    for k in L: x = L[k].to(torch.bfloat16); y = S[k].to(torch.bfloat16); same += (x == y).sum().item(); tot += x.numel()
    say(f"t={t}: fração de elementos IDÊNTICOS após o cast para bf16 (SLERP vs LERP): {same/tot:.4f}")
    # distância bf16 relativa
    num = den = 0.0
    for k in L: x = L[k].to(torch.bfloat16).float(); y = S[k].to(torch.bfloat16).float(); num += ((x - y) ** 2).sum().item(); den += (x ** 2).sum().item()
    say(f"t={t}: distância L2 relativa em bf16: {(num/den)**.5:.2e}")
tok = AutoTokenizer.from_pretrained(mp("qwen25-05b")); m = AutoModelForCausalLM.from_pretrained(mp("qwen25-05b"), torch_dtype=torch.float32).to(E.DEV).eval()
for t in (0.4, 0.6):
    for name, st in (("LERP", M.lerp(b, i, t)), ("SLERP", M.slerp(b, i, t))):
        M.load_into(m, st, torch.float32); r = E.eval_suite(m, tok, E.TASKS, "val"); np.savez(f"{ROOT}/cache/fp32_q_{name.lower()}_{t}_val.npz", **r); say(f"fp32 val macro {name} t={t}: {mac(r):.4f}")
open(f"{ROOT}/results/slerp_bf16_check.txt", "w").write("\n".join(out) + "\n")
