"""Sensibilidade à precisão: reavalia em fp32 (a) Pythia: 6 checkpoints isolados + médias k=1..6 (NLL por bloco, 140 blocos);
(b) Qwen base<->instruct LERP: NLL por bloco (blocos 0-139) em t=0..1 e acurácia de validação em t=0,0.7,1.
Saídas: cache/fp32_*.npz e results/fp32_check.txt. Uso: python3 experiments/fp32_check.py [pythia|qwen]"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import ROOT, mp, ppl_chunks, mac, W, M, E
from transformers import AutoModelForCausalLM, AutoTokenizer
def host(name):
    tok = AutoTokenizer.from_pretrained(mp(name)); m = AutoModelForCausalLM.from_pretrained(mp(name), torch_dtype=torch.float32).to(E.DEV).eval(); return m, tok
def save(k, **kw): np.savez(f"{ROOT}/cache/{k}.npz", **kw)
LOG = []
def say(s): print(s, flush=True); LOG.append(s)
which = sys.argv[1:] or ["pythia", "qwen"]
if "pythia" in which:
    steps = [143000, 140000, 130000, 120000, 110000, 100000]; names = [f"pythia160m-step{s}" for s in steps]; m, tok = host(names[0]); sds = [W(n) for n in names]
    for s, n, sd in zip(steps, names, sds):
        M.load_into(m, sd, torch.float32); x = ppl_chunks(m, tok, 0, 140); save(f"fp32_p_single{s}", x=x); say(f"fp32 single {s}: val(0-39) {x[:40].mean():.4f} test(40-139) {x[40:].mean():.4f}")
    for k in range(1, 7):
        M.load_into(m, M.average(sds[:k]), torch.float32); x = ppl_chunks(m, tok, 0, 140); save(f"fp32_p_avg{k}", x=x); say(f"fp32 avg k={k}: val {x[:40].mean():.4f} test {x[40:].mean():.4f}")
if "qwen" in which:
    b, i = W("qwen25-05b"), W("qwen25-05b-instruct"); m, tok = host("qwen25-05b")
    for t in np.round(np.arange(0, 1.01, .1), 1):
        M.load_into(m, M.lerp(b, i, float(t)), torch.float32); x = ppl_chunks(m, tok, 0, 140); save(f"fp32_q_lerp_{t:.1f}_nll", x=x); say(f"fp32 qwen LERP t={t:.1f}: NLL val {x[:40].mean():.4f} test {x[40:].mean():.4f}")
    for t in (0.0, 0.7, 1.0):
        M.load_into(m, M.lerp(b, i, float(t)), torch.float32); t0 = time.time(); r = E.eval_suite(m, tok, E.TASKS, "val"); save(f"fp32_q_lerp_{t:.1f}_val", **r); say(f"fp32 qwen LERP t={t:.1f}: val macro {mac(r):.4f} ({time.time()-t0:.0f}s)")
open(f"{ROOT}/results/fp32_check_{'_'.join(which)}.txt", "w").write("\n".join(LOG) + "\n")
