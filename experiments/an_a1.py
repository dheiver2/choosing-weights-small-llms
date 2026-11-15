import sys, os, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import *; sys.path.insert(0, ROOT); from common import stats as S
T = [f"{t:.1f}" for t in np.arange(0, 1.01, .1)]
ld = lambda k, sp: {t: np.load(key_path(k, sp))[t] for t in E.TASKS}
for t in T:
    v, te = ld(f"q_lerp_{t}", "val"), ld(f"q_lerp_{t}", "test"); print(t, round(mac(v), 4), round(mac(te), 4))
