import numpy as np
from cogcanvas.env import CanvasEnv
from cogcanvas.metrics import f1_score
from cogcanvas.baselines import random_policy

e = CanvasEnv(seed=0)
obs = e.reset()
rng = np.random.default_rng(0)

print(f"initial: prev_potential={e.prev_potential}, F1={f1_score(e.canvas, e.target):.3f}")
print()

total = 0.0
step = 0
done = False
while not done and step < 24:
    a = random_policy(obs, e, rng)
    obs, r, done, info = e.step(a)
    step += 1
    total += r
    f1 = f1_score(e.canvas, e.target)
    print(f"step {step:2d}: a={a:3d}  r={r:+.3f}  F1={f1:.3f}  "
          f"prev_pot={e.prev_potential:.3f}  done={done}")
print(f"\ntotal reward = {total:+.3f}")
print(f"final F1 = {f1_score(e.canvas, e.target):.3f}")