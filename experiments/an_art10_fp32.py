"""Artigo 10: sensibilidade a precisão (bf16 vs fp32) do NLL por t e da acurácia de validação (t=0,0.7,1).
Saída: results/summary_art10_fp32.txt"""
import os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = lambda k: np.load(f"{ROOT}/cache/{k}.npz")["x"]
out = []; say = lambda s: (out.append(s), print(s))
say("NLL por token (nats), validação = blocos 0-39, teste = 40-139; bf16 | fp32 | diferença bf16-fp32")
dv, dt = [], []; fv, ft = {}, {}
for i in range(11):
    t = f"{i/10:.1f}"; f = L(f"fp32_q_lerp_{t}_nll"); bv = L(f"q_lerp_{t}__ppl0_40"); bt = L(f"q_lerp_{t}__ppl40_100")
    fv[t], ft[t] = f[:40].mean(), f[40:].mean(); dv.append(bv.mean()-fv[t]); dt.append(bt.mean()-ft[t])
    say(f"t={t}: val {bv.mean():.4f} | {fv[t]:.4f} | {dv[-1]:+.4f}; test {bt.mean():.4f} | {ft[t]:.4f} | {dt[-1]:+.4f}")
say(f"diferença bf16-fp32 teste: min {min(dt):+.4f} max {max(dt):+.4f}; validação: min {min(dv):+.4f} max {max(dv):+.4f}")
say(f"custo teste t=1 vs t=0: fp32 {ft['1.0']-ft['0.0']:+.4f}; bf16 {L('q_lerp_1.0__ppl40_100').mean()-L('q_lerp_0.0__ppl40_100').mean():+.4f}")
a = min(fv, key=fv.get); say(f"argmin NLL validação fp32: t={a} ({fv[a]:.4f}); t=0.1: {fv['0.1']:.4f}; gap {fv['0.1']-fv[a]:.4f}")
bvv = {t: np.mean([np.load(f"{ROOT}/cache/q_lerp_{t}__val.npz")[k].mean() for k in ["arc_easy","arc_challenge","hellaswag","winogrande"]]) for t in ["0.0","0.7","1.0"]}
for t in ["0.0","0.7","1.0"]:
    d = np.load(f"{ROOT}/cache/fp32_q_lerp_{t}_val.npz"); m = np.mean([d[k].mean() for k in d.files])
    say(f"val macro t={t}: bf16 {bvv[t]:.4f} | fp32 {m:.4f}")
open(f"{ROOT}/results/summary_art10_fp32.txt", "w").write("\n".join(out) + "\n")
