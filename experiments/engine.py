"""Motor: carrega pesos, monta fusões, avalia com cache por item em cache/<chave>__<split>.npz."""
import os, sys, json, time, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from common import evalkit as E, mergeops as M
CACHE = os.path.join(ROOT, "cache"); os.makedirs(CACHE, exist_ok=True)
def mp(n): return os.path.join(ROOT, "models", n)
_state = {}
def get(family_model):
    """Modelo/tokenizer residentes por família (arquitetura igual dentro da família)."""
    if family_model not in _state:
        m, t = E.load(mp(family_model)); _state[family_model] = (m, t)
    return _state[family_model]
_sd = {}
def W(n):
    if n not in _sd: _sd[n] = M.sd(mp(n))
    return _sd[n]
def key_path(key, split): return os.path.join(CACHE, key.replace("/", "_") + f"__{split}.npz")
def have(key, split): return os.path.exists(key_path(key, split))
def ev(key, state, host, split, tasks=E.TASKS):
    """Avalia o estado de pesos `state` no modelo `host`; devolve dict tarefa->vetor 0/1."""
    p = key_path(key, split)
    if os.path.exists(p):
        z = np.load(p); return {t: z[t] for t in tasks}
    m, t = get(host); M.load_into(m, state); t0 = time.time()
    r = E.eval_suite(m, t, tasks, split); np.savez(p, **r)
    print(f"[{time.strftime('%H:%M:%S')}] {key} {split} macro={E.macro(r):.4f} ({time.time()-t0:.0f}s)", flush=True); return r
def evppl(key, state, host, start, n):
    p = key_path(key, f"ppl{start}_{n}")
    if os.path.exists(p): return np.load(p)["x"]
    m, t = get(host); M.load_into(m, state)
    x = ppl_chunks(m, t, start, n); np.savez(p, x=x); print(f"[{time.strftime('%H:%M:%S')}] {key} ppl{start}_{n} mean={x.mean():.4f}", flush=True); return x
@torch.no_grad()
def ppl_chunks(model, tok, start, n, L=512):
    from datasets import load_from_disk
    d = load_from_disk(os.path.join(ROOT, "data", "wikitext2"))["test"]
    ids = tok("\n\n".join(d["text"]), add_special_tokens=False, return_tensors="pt")["input_ids"][0]; out = []
    for k in range(start, start + n):
        x = ids[k * L:(k + 1) * L + 1]
        if len(x) < L + 1: break
        lg = model(input_ids=x[:-1].unsqueeze(0).to(E.DEV)).logits.float()[0]
        out.append(float(torch.nn.functional.cross_entropy(lg, x[1:].to(E.DEV))))
    return np.array(out)
def mac(r): return float(np.mean([v.mean() for v in r.values()]))
