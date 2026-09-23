import gymnasium as gym 
import numpy as np 
from gymnasium import spaces
from cogcanvas.env import CanvasEnv

class GymCanvasEnv(gym.Env):
    """ Gymnasium adapter for CanvasEnv."""

    metadata = {"render_modes": []}

    def __init__(self, **env_kwargs):
        super().__init__()
        self.env = CanvasEnv(**env_kwargs)
        self.observation_space = spaces.Box(
            low = 0.0, high = 1.0, 
            shape = (2, self.env.size, self.env.size),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(self.env.n_actions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(int(action))
        return obs, reward, done, False, info