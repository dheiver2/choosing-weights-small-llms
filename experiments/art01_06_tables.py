"""Tabelas dos artigos 1 e 6 (LERP Qwen / SmolLM2): val e teste com IC, diferença pareada vs melhor original, por tarefa. Saída: artigos/tab_art01.tex, tab_art06.tex, results/art01_06_extra.txt"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import key_path, ROOT, mac
sys.path.insert(0, ROOT); from common import stats as S, evalkit as E
TASKS = E.TASKS
def ld(k, sp): z = np.load(key_path(k, sp)); return {t: z[t] for t in TASKS}
def ci(res, B=2000):
    rng = np.random.default_rng(0); ms = [np.mean([r[rng.integers(0, len(r), len(r))].mean() for r in res.values()]) for _ in range(B)]
    return mac(res), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
def pdiff(a, b): return S.macro_paired(a, b, B=3000)
def run(prefix, ts, art, orig="0.0"):
    val = {t: ld(f"{prefix}_{t}", "val") for t in ts}; te = {t: ld(f"{prefix}_{t}", "test") for t in ts}
    tstar = max(ts, key=lambda t: mac(val[t])); lines = []; rows = []
    for t in ts:
        v = ci(val[t]); s = ci(te[t]); d = pdiff(te[t], te[orig])
        rows.append((t, v, s, d))
        lines.append(f"{t} val {v[0]:.4f} [{v[1]:.4f},{v[2]:.4f}] test {s[0]:.4f} [{s[1]:.4f},{s[2]:.4f}] diff_vs_{orig} {d[0]*100:+.2f} [{d[1]*100:+.2f},{d[2]*100:+.2f}] p={d[3]:.3f}")
    # por tarefa em t*, base e instruct
    lines.append(f"t*={tstar}")
    for t in (orig, tstar, ts[-1]):
        lines.append(f"per-task test t={t}: " + ", ".join(f"{k}={te[t][k].mean():.4f}(n={len(te[t][k])})" for k in TASKS))
        lines.append(f"per-task val  t={t}: " + ", ".join(f"{k}={val[t][k].mean():.4f}(n={len(val[t][k])})" for k in TASKS))
    d_inst = pdiff(te[ts[-1]], te[orig]); lines.append(f"instruct - base test: {d_inst[0]*100:+.2f} [{d_inst[1]*100:+.2f},{d_inst[2]*100:+.2f}] p={d_inst[3]:.3f}")
    vi = pdiff(val[ts[-1]], val[orig]); lines.append(f"instruct - base val: {vi[0]*100:+.2f} [{vi[1]*100:+.2f},{vi[2]*100:+.2f}] p={vi[3]:.3f}")
    # contagem de itens (concordância base/instruct)
    best_test = max(ts, key=lambda t: mac(te[t])); lines.append(f"oracle test t={best_test}")
    # linhas de tabela LaTeX
    tex = []
    for t, v, s, d in rows:
        mark = r"\textbf{%s}" % t if t == tstar else t
        tex.append(f"{mark} & {v[0]*100:.2f} [{v[1]*100:.2f}, {v[2]*100:.2f}] & {s[0]*100:.2f} [{s[1]*100:.2f}, {s[2]*100:.2f}] & " + ("--" if t == orig else f"{d[0]*100:+.2f} [{d[1]*100:+.2f}, {d[2]*100:+.2f}]") + " & " + ("--" if t == orig else ("$<$0.001" if d[3] < 0.0005 else f"{d[3]:.3f}")) + r" \\")
    open(os.path.join(ROOT, "artigos", f"tab_art{art}.tex"), "w").write("\n".join(tex) + "\n\\bottomrule\n")
    return lines
L = ["== Qwen =="] + run("q_lerp", [f"{t:.1f}" for t in np.arange(0, 1.01, .1)], "01")
L += ["== SmolLM2 =="] + run("s_lerp", ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], "06")
open(os.path.join(ROOT, "results", "art01_06_extra.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
