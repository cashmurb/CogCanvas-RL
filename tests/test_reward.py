import numpy as np
from cogcanvas.env import CanvasEnv
from cogcanvas.metrics import f1_score, clutter, abstraction

# metric tests

def test_f1_blank_is_zero():
    canvas = np.zeros((8, 8), dtype=np.int8)
    target = np.zeros((8, 8), dtype=np.int8)
    target[2, 3] = target[5, 1] = target[6, 6] = 1
    assert f1_score(canvas, target) == 0.0


def test_f1_perfect_is_one():
    target = np.zeros((8, 8), dtype=np.int8)
    target[2, 3] = target[5, 1] = target[6, 6] = 1
    canvas = target.copy()
    assert abs(f1_score(canvas, target) - 1.0) < 1e-9


def test_f1_extra_marks_hurts():
    target = np.zeros((8, 8), dtype=np.int8)
    target[2, 3] = target[5, 1] = target[6, 6] = 1
    canvas = target.copy()
    canvas[0, 0] = 1                      
    assert f1_score(canvas, target) < 1.0


def test_clutter_blank_is_zero():
    canvas = np.zeros((8, 8), dtype=np.int8)
    assert clutter(canvas, cap=6) == 0.0


def test_clutter_rises_with_marks():
    blank = np.zeros((8, 8), dtype=np.int8)
    some  = blank.copy(); some[2, 3] = some[5, 1] = some[6, 6] = 1
    many  = blank.copy()
    for i in range(20):
        many.flat[i] = 1
    assert clutter(blank, 6) < clutter(some, 6) < clutter(many, 6)

# reward behaviour tests 
def test_lazy_policy_negative():
    """STOP immediately. Should score negative."""
    e = CanvasEnv(seed=0); e.reset()
    _, r, _, _ = e.step(2 * e.n_cells)
    assert r < 0


def test_filled_policy_negative():
    """Fill every cell, then STOP. Should score negative."""
    e = CanvasEnv(seed=0, max_steps=100); e.reset()
    total = 0.0
    for i in range(e.n_cells):
        _, r, d, _ = e.step(i)
        total += r
        if d: break
    if not d:
        _, r, d, _ = e.step(2 * e.n_cells)
        total += r
    assert total < 0


def test_perfect_policy_positive():
    """Place marks on exactly the target cells, then STOP."""
    e = CanvasEnv(seed=0); e.reset()
    total = 0.0
    for y in range(e.size):
        for x in range(e.size):
            if e.target[y, x] == 1:
                _, r, _, _ = e.step(y * e.size + x)
                total += r
    _, r, _, _ = e.step(2 * e.n_cells)
    total += r
    assert total > 0

def test_perfect_beats_blank_and_filled():
    """Perfect > blank > filled."""
    def run(env, actions):
        total = 0.0
        for a in actions:
            _, r, d, _ = env.step(a)
            total += r
            if d: break
        return total

    # blank
    e1 = CanvasEnv(seed=0); e1.reset()
    blank = run(e1, [2 * e1.n_cells])

    # filled
    e2 = CanvasEnv(seed=0, max_steps=100); e2.reset()
    filled = run(e2, list(range(e2.n_cells)) + [2 * e2.n_cells])

    # perfect
    e3 = CanvasEnv(seed=0); e3.reset()
    perfect_actions = [y * e3.size + x
                       for y in range(e3.size)
                       for x in range(e3.size)
                       if e3.target[y, x] == 1]
    perfect_actions.append(2 * e3.n_cells)
    perfect = run(e3, perfect_actions)

    assert perfect > blank > filled