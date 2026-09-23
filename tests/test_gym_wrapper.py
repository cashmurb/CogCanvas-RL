import numpy as np
from cogcanvas.gym_wrapper import GymCanvasEnv


def test_wrapper_reset_returns_obs_info():
    env = GymCanvasEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (2, 8, 8)
    assert isinstance(info, dict)


def test_wrapper_step_returns_five_tuple():
    env = GymCanvasEnv(seed=0)
    env.reset()
    result = env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (2, 8, 8)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_wrapper_action_space():
    env = GymCanvasEnv(seed=0)
    assert env.action_space.n == 2 * 8 * 8 + 1