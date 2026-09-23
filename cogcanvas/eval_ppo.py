import numpy as np
from stable_baselines3 import PPO
from cogcanvas.gym_wrapper import GymCanvasEnv

def evaluate(model_path, n_episodes=50, seed=100, deterministic=True):
    model = PPO.load(model_path)
    rewards = []
    matches = []
    lengths = []

    for ep in range(n_episodes):
        env = GymCanvasEnv(seed=seed + ep)
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        info = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, _, info = env.step(action)
            total += r
            steps += 1
        rewards.append(total)
        matches.append(info["match"])
        lengths.append(steps)
    
    rewards = np.array(rewards)
    matches = np.array(matches)
    lengths = np.array(lengths)

    print(f"Episodes: {n_episodes}")
    print (f"Mean reward: {rewards.mean():+.3f} (±{rewards.std():.3f})")
    print(f"Mean match (F1): {matches.mean():.3f}")
    print(f"Mean steps: {lengths.mean():.1f}")
    print(f"Success rate (F1 > 0.9): {(matches > 0.9).mean():.1%}")

if __name__ == "__main__":
    evaluate("runs/ppo_rung0/final_model")
    