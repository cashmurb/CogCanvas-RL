import numpy as np 
from tensorboardX import SummaryWriter

class TBLogger:
    """
    Tensorboard logger for RL training. Tracks 
    - episode rewards (raw + rolling mean)
    - episode stats: steps, match, clutter, abstraction
    - reward components: task, clutter_bonus, abstraction_bonus, step_cost
    - training scalars (optional): loss, policy entroy
    """

    def __init__(self, log_dir="runs/cogcanvas", window=100):
        self.writer = SummaryWriter(log_dir)
        self.window = window
        self.reward_history = []
        self.step_count = 0
    
    def log_episode(self, episode, total_reward, steps, info):
        self.reward_history.append(total_reward)
        self.writer.add_scalar("reward/raw", total_reward, episode)

        if len(self.reward_history) >= self.window:
            rolling = float(np.mean(self.reward_history[-self.window:]))
            self.writer.add_scalar("reward/rolling_mean", rolling, episode)
        
        self.writer.add_scalar("episode/steps", steps, episode)
        self.writer.add_scalar("episode/match", info.get("match", 0.0), episode)
        self.writer.add_scalar("episode/clutter", info.get("clutter", 0.0), episode)
        self.writer.add_scalar("episode/abstraction", info.get("abstraction", 0.0), episode)
        self.writer.add_scalar("reward_components/task", info.get("task", 0.0), episode)
        self.writer.add_scalar("reward_components/clutter_bonus", info.get("clutter_bonus", 0.0), episode)
        self.writer.add_scalar("reward_components/abstraction_bonus", info.get("abstraction_bonus", 0.0), episode)
        self.writer.add_scalar("reward_components/step_cost", info.get("step_cost", 0.0), episode)

    def log_training_step(self, loss=None, policy_entropy=None):
        self.step_count += 1
        if loss is not None:
            self.writer.add_scalar("training/loss", loss, self.step_count)
        if policy_entropy is not None:
            self.writer.add_scalar("training/policy_entropy", policy_entropy, self.step_count)

    def close(self):
        self.writer.close()