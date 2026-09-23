import numpy as np
from cogcanvas.env import CanvasEnv


def test_determinism():
    o1 = CanvasEnv(seed=0).reset()
    o2 = CanvasEnv(seed=0).reset()
    assert np.array_equal(o1, o2)


def test_place_marks_cell():
    e = CanvasEnv(seed=0); e.reset()
    e.step(0)                          
    assert e.canvas[0, 0] == 1


def test_erase_clears_cell():
    e = CanvasEnv(seed=0); e.reset()
    e.step(0)                         
    e.step(64)                         
    assert e.canvas[0, 0] == 0


def test_stop_ends_episode():
    e = CanvasEnv(seed=0); e.reset()
    _, _, done, _ = e.step(128)        
    assert done is True


def test_max_steps_ends_episode():
    e = CanvasEnv(seed=0); e.reset()
    done = False
    for _ in range(24):
        _, _, done, _ = e.step(0)
    assert done is True


def test_observation_shape():
    e = CanvasEnv(size=8, seed=0)
    obs = e.reset()
    assert obs.shape == (2, 8, 8)
    assert obs.dtype == np.float32