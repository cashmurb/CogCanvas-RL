from cogcanvas.env import CanvasEnv
from cogcanvas.prm import HeuristicPRM
from cogcanvas.baselines import random_policy, evaluate

# 1. What does the default env look like?
e = CanvasEnv()
print("env.prm:", e.prm)
print("env.w_shaping:", e.w_shaping)
print("env.w_step:", e.w_step)
print("env.max_steps:", e.max_steps)

# 2. Random with default env
s = evaluate(random_policy, n_episodes=20, seed=0)
print("\nrandom (default):", s["reward_mean"])

# 3. Random with explicit PRM
s = evaluate(random_policy, n_episodes=20, seed=0, prm=HeuristicPRM())
print("random (with PRM):", s["reward_mean"])

# 4. Random with explicit no-PRM
s = evaluate(random_policy, n_episodes=20, seed=0, prm=None)
print("random (no PRM):", s["reward_mean"])