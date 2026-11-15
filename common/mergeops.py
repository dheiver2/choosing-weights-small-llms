"""Operações no espaço de pesos sobre state_dicts em fp32 (CPU). Todas devolvem um novo dict."""
import os, re, glob
import numpy as np, torch
from safetensors.torch import load_file

def sd(path):
    """Carrega todos os .safetensors de um diretório em fp32."""
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))): out.update({k: v.float() for k, v in load_file(f).items()})
    return out

def layer_of(name):
    m = re.search(r"layers?\.(\d+)\.", name); return int(m.group(1)) if m else None

def lerp(a, b, t):            # (1-t)*a + t*b
    return {k: (1 - t) * a[k] + t * b[k] for k in a}

def slerp(a, b, t, eps=1e-8):
    """SLERP por tensor (vetor achatado); cai para LERP se quase colineares."""
    out = {}
    for k in a:
        x, y = a[k].flatten(), b[k].flatten(); nx, ny = x.norm(), y.norm()
        if nx < eps or ny < eps: out[k] = (1 - t) * a[k] + t * b[k]; continue
        cos = torch.clamp((x / nx) @ (y / ny), -1, 1); om = torch.acos(cos)
        if om.abs() < 1e-4: out[k] = (1 - t) * a[k] + t * b[k]; continue
        so = torch.sin(om); out[k] = ((torch.sin((1 - t) * om) / so) * x + (torch.sin(t * om) / so) * y).view_as(a[k])
    return out

def delta(a, base):           # a - base
    return {k: a[k] - base[k] for k in a}

def add(base, d, lam=1.0):    # base + lam*d  (lam pode ser float ou dict por camada via callable)
    return {k: base[k] + (lam(k) if callable(lam) else lam) * d[k] for k in base}

def average(sds, w=None):
    n = len(sds); w = w or [1.0 / n] * n
    return {k: sum(wi * s[k] for wi, s in zip(w, sds)) for k in sds[0]}

def dare(d, p, seed=0):
    """Descarta fração p dos elementos do delta e reescala por 1/(1-p)."""
    g = torch.Generator().manual_seed(seed); out = {}
    for k, v in d.items():
        m = (torch.rand(v.shape, generator=g) >= p).float(); out[k] = v * m / (1 - p)
    return out

def ties(ds, density=0.2, lam=1.0):
    """TIES: poda por magnitude (mantém `density`), elege sinal por massa, média disjunta dos que concordam."""
    out = {}
    for k in ds[0]:
        ts = []
        for d in ds:
            v = d[k].flatten(); kk = max(1, int(density * v.numel())); thr = v.abs().kthvalue(v.numel() - kk + 1).values
            ts.append(torch.where(v.abs() >= thr, v, torch.zeros_like(v)))
        st = torch.stack(ts); sign = torch.sign(st.sum(0)); agree = (torch.sign(st) == sign) & (st != 0)
        cnt = agree.sum(0).clamp(min=1); out[k] = (lam * (st * agree).sum(0) / cnt).view_as(ds[0][k])
    return out

def cosine(d1, d2):
    x = torch.cat([v.flatten() for v in d1.values()]); y = torch.cat([v.flatten() for v in d2.values()])
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))

def l2(d): return float(torch.sqrt(sum((v ** 2).sum() for v in d.values())))

def load_into(model, state, dtype=torch.bfloat16):
    """Copia state (fp32) para o modelo já na GPU, preservando pesos amarrados."""
    own = model.state_dict()
    for k, v in state.items():
        if k in own: own[k].copy_(v.to(own[k].device, own[k].dtype))
    return model
