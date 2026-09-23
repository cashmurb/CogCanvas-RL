import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from cogcanvas.gym_wrapper import GymCanvasEnv


class ProgressCallback(BaseCallback):
    """Print a concise progress line every `log_every` steps."""

    def __init__(self, log_every=5_000, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every
        self.last_logged = 0

    def _on_step(self):
        if self.num_timesteps - self.last_logged >= self.log_every:
            self.last_logged = self.num_timesteps

            buf = self.model.ep_info_buffer
            if buf:
                rew = np.mean([e["r"] for e in buf])
                len_ = np.mean([e["l"] for e in buf])
                print(f"[step {self.num_timesteps:>7}] "
                      f"ep_rew_mean={rew:+.2f}  ep_len_mean={len_:.1f}")
            else:
                print(f"[step {self.num_timesteps:>7}] no episodes yet")

            self.logger.record("training/step", self.num_timesteps)

        return True


def train(total_timesteps=500_000, log_dir="runs/ppo_rung0", seed=0):
    env = Monitor(GymCanvasEnv(seed=seed, max_steps=12))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=128,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )

    callback = ProgressCallback(log_every=5_000)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(f"{log_dir}/final_model")
    print(f"Training complete. Model saved to {log_dir}/final_model")
    return model


if __name__ == "__main__":
    train()