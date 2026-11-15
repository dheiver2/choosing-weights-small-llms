import numpy as np
def boot_ci(x, B=5000, seed=0, a=0.05):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(x.mean()), float(np.percentile(m, 100 * a / 2)), float(np.percentile(m, 100 * (1 - a / 2)))
def paired_diff(a, b, B=5000, seed=0):
    """Diferença média a-b com IC 95% bootstrap pareado por item; p (bicaudal) por bootstrap."""
    d = np.asarray(a, float) - np.asarray(b, float); rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(min(p, 1.0))
def macro_paired(res_a, res_b, **kw):
    """Diferença pareada na média macro das tarefas: concatena a diferença por item ponderando cada tarefa igualmente."""
    ds = [np.asarray(res_a[t], float) - np.asarray(res_b[t], float) for t in res_a]
    rng = np.random.default_rng(kw.get("seed", 0)); B = kw.get("B", 5000); ms = []
    for _ in range(B):
        ms.append(np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in ds]))
    ms = np.array(ms); mean = float(np.mean([d.mean() for d in ds]))
    return mean, float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)), float(min(2 * min((ms <= 0).mean(), (ms >= 0).mean()), 1.0))
