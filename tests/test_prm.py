import numpy as np
from cogcanvas.prm import HeuristicPRM
from cogcanvas.env import CanvasEnv


def _obs(canvas, target):
    return np.stack([canvas, target]).astype(np.float32)


def test_prm_returns_scalar():
    prm = HeuristicPRM()
    obs = _obs(np.zeros((8, 8), np.int8), np.zeros((8, 8), np.int8))
    s = prm.score(obs)
    assert isinstance(s, float)


def test_prm_range():
    prm = HeuristicPRM()
    obs = _obs(np.zeros((8, 8), np.int8), np.zeros((8, 8), np.int8))
    assert 0.0 <= prm.score(obs) <= 1.0


def test_prm_blank_is_low():
    prm = HeuristicPRM()
    canvas = np.zeros((8, 8), np.int8)
    target = np.zeros((8, 8), np.int8)
    target[2, 3] = target[5, 1] = target[6, 6] = 1
    s_blank = prm.score(_obs(canvas, target))
    assert s_blank < 0.4


def test_prm_perfect_is_high():
    prm = HeuristicPRM()
    target = np.zeros((8, 8), np.int8)
    target[2, 3] = target[5, 1] = target[6, 6] = 1
    canvas = target.copy()
    s_perfect = prm.score(_obs(canvas, target))
    assert s_perfect > 0.8


def test_prm_monotone_in_progress():
    prm = HeuristicPRM()
    target = np.zeros((8, 8), np.int8)
    target[2, 3] = target[5, 1] = target[6, 6] = 1

    canvas = np.zeros((8, 8), np.int8)
    s0 = prm.score(_obs(canvas, target))

    canvas[2, 3] = 1
    s1 = prm.score(_obs(canvas, target))

    canvas[5, 1] = 1
    s2 = prm.score(_obs(canvas, target))

    canvas[6, 6] = 1
    s3 = prm.score(_obs(canvas, target))

    assert s0 < s1 < s2 < s3


def test_env_accepts_prm():
    prm = HeuristicPRM()
    env = CanvasEnv(seed=0, prm=prm)
    env.reset()
    obs, r, d, info = env.step(0)
    assert isinstance(r, float)
    assert "prm_score" in info  # new key