import numpy as np
from cogcanvas.env import CanvasEnv
from cogcanvas.prm import HeuristicPRM
from cogcanvas.baselines import random_policy, scripted_policy


def _rollout(env, policy, rng, prm, X_list, y_list):
    obs = env.reset()
    X_list.append(obs)
    y_list.append(prm.score(obs))

    done = False
    while not done:
        action = policy(obs, env, rng)
        obs, _, done, _ = env.step(action)
        X_list.append(obs)
        y_list.append(prm.score(obs))

def generate_dataset(n_episodes=500, seed=0):
    """
    Mix of random and scripted rollouts.
    - Random: covers low-F1 states
    - Scripted: covers high-F1 states

    This ensures the training data spans the whole score range.
    """
    rng = np.random.default_rng(seed)
    prm = HeuristicPRM()
    env = CanvasEnv(seed=seed)

    X_list, y_list = [], []

    # 60% random — covers blank and messy states
    n_random = int(n_episodes * 0.6)
    for ep in range(n_random):
        _rollout(env, random_policy, rng, prm, X_list, y_list)

    # 40% scripted — covers correct and near-correct states
    n_scripted = n_episodes - n_random
    for ep in range(n_scripted):
        _rollout(env, scripted_policy, rng, prm, X_list, y_list)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def split_dataset(X, y, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]