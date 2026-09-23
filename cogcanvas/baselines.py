import numpy as np
from cogcanvas.env import CanvasEnv

def lazy_policy(obs, env, rng):
    """STOP immediately."""
    return 2 * env.n_cells

def random_policy(obs, env, rng):
    """Randomly choose an action."""
    return int(rng.integers(env.n_actions))

def scripted_policy(obs, env,rng):
    """Place a mark on each target cell, then STOP."""
    target = obs[1]
    canvas = obs[0]

    # find first target cell that is not yet marked
    for y in range(env.size):
        for x in range(env.size):
            if target[y,x] == 1 and canvas[y,x] == 0:
                return y * env.size + x
    
    # every target cell is marked, so STOP
    return 2 * env.n_cells

# runner 

def run_episode(env, policy, rng, max_steps=None):
    """Run a single episode in the environment using the given policy."""
    obs = env.reset()
    done = False 
    total = 0.0
    steps = 0
    info = {}

    while not done:
        action = policy(obs, env, rng)
        obs, r, done, info = env.step(action)
        total += r
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    
    return total, steps, info

def evaluate(policy, n_episodes=100, seed=None, **env_kwargs):
    """Run n episodes of the environment using the given policy and return the average reward."""
    rng = np.random.default_rng(seed)
    rewards = []
    steps_list = []
    matches = []

    for ep in range(n_episodes):
        env = CanvasEnv(seed=seed + ep, **env_kwargs)
        total, steps, info = run_episode(env, policy, rng)
        rewards.append(total)
        steps_list.append(steps)
        matches.append(info.get("match", 0.0))
    
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "steps_mean": float(np.mean(steps_list)),
        "match_mean": float(np.mean(matches)),
    }
