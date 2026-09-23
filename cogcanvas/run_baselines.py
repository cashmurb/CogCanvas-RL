from cogcanvas.env import CanvasEnv
from cogcanvas.baselines import random_policy
from cogcanvas.logger import TBLogger
import numpy as np

def run(n_episodes=50, seed=0, log_dir="runs/phase4_random"):
    rng = np.random.default_rng(seed)
    logger = TBLogger(log_dir=log_dir, window=20)

    for ep in range(n_episodes):
        env = CanvasEnv(seed=seed + ep)
        obs = env.reset()
        done = False
        total = 0.0
        steps = 0

        while not done:
            a = random_policy(obs, env, rng)
            obs, r, done, info = env.step(a)
            total += r
            steps += 1

        logger.log_episode(ep, total, steps, info)

    logger.close()
    print(f"Done. Logs in {log_dir}")


if __name__ == "__main__":
    run()