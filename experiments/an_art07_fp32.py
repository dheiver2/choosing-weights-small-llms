"""Artigo 7 em fp32 (corrige o artefato de bf16): tabela de NLL por checkpoint isolado e por média, diferenças pareadas por bloco, seleção por validação.
Saída: results/summary_art07_fp32.txt"""
import sys, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from common import stats as S
L = lambda k: np.load(f"{ROOT}/cache/{k}.npz")["x"]; steps = [143000, 140000, 130000, 120000, 110000, 100000]
sg = {s: L(f"fp32_p_single{s}") for s in steps}; av = {k: L(f"fp32_p_avg{k}") for k in range(1, 7)}; V = lambda x: x[:40]; T = lambda x: x[40:]
out = []; say = lambda s: (out.append(s), print(s))
say("fp32, 40 blocos de validação (0-39) e 100 de teste (40-139); NLL médio por token (nats)")
for s in steps: say(f"single {s//1000}k: val {V(sg[s]).mean():.4f} test {T(sg[s]).mean():.4f}")
for k in range(1, 7): say(f"avg k={k} (143k..{steps[k-1]//1000}k): val {V(av[k]).mean():.4f} test {T(av[k]).mean():.4f}")
cands = {**{f"single_{s}": sg[s] for s in steps}, **{f"avg_k{k}": av[k] for k in range(2, 7)}}
bv = min(cands, key=lambda c: V(cands[c]).mean()); ba = min((f"avg_k{k}" for k in range(2, 7)), key=lambda c: V(cands[c]).mean())
say(f"melhor candidato na validação entre 6 singles + médias k=2..6: {bv}; melhor média na validação: {ba}")
pd = lambda a, b: S.paired_diff(T(a), T(b))
for k in range(2, 7): d = pd(av[k], av[1]); say(f"avg k={k} - último (143k), teste: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] p={d[3]:.3f}")
for s in steps[1:]: d = pd(sg[s], sg[143000]); say(f"single {s//1000}k - último, teste: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] p={d[3]:.3f}")
bs = min(steps, key=lambda s: V(sg[s]).mean()); say(f"melhor single na validação (baseline pré-registrado, melhor original): {bs//1000}k")
d = pd(av[int(ba[-1])], sg[bs]); say(f"{ba} - melhor single ({bs//1000}k), teste: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] p={d[3]:.3f}")
d = pd(cands[bv], sg[143000]); say(f"candidato escolhido na validação ({bv}) - último, teste: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] p={d[3]:.3f}")
say("--- cada média contra o melhor MEMBRO dela (melhor single do conjunto por NLL de teste; descritivo) ---")
for k in range(2, 6):
    mem = steps[:k]; bm = min(mem, key=lambda s_: T(sg[s_]).mean()); d = pd(av[k], sg[bm]); say(f"avg k={k} - single {bm//1000}k (melhor membro), teste: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] p={d[3]:.3f}")
say("perplexidade (exp NLL, teste): último " + f"{np.exp(T(sg[143000]).mean()):.1f}; melhor single {np.exp(T(sg[bs]).mean()):.1f}; avg5 {np.exp(T(av[5]).mean()):.1f}")
say("--- sensibilidade à precisão: o mesmo cálculo em bf16 (artefato) ---")
for k in range(1, 7):
    b = np.load(f"{ROOT}/cache/p_avg{k}__ppl40_100.npz")["x"]; say(f"bf16 avg k={k}: test {b.mean():.4f} (fp32: {T(av[k]).mean():.4f}); diferença bf16-fp32 {b.mean()-T(av[k]).mean():+.4f}")
open(f"{ROOT}/results/summary_art07_fp32.txt", "w").write("\n".join(out) + "\n")
