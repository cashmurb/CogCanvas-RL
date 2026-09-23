from cogcanvas.baselines import (lazy_policy, random_policy, scripted_policy, run_episode, evaluate)

def test_lazy_is_negative():
    stats = evaluate(lazy_policy, n_episodes=50, seed=0)
    assert stats["reward_mean"] < 0


def test_random_is_negative():
    stats = evaluate(random_policy, n_episodes=50, seed=0)
    assert stats["reward_mean"] < 0


def test_scripted_is_positive():
    stats = evaluate(scripted_policy, n_episodes=50, seed=0)
    assert stats["reward_mean"] > 0


def test_ordering():
    lazy = evaluate(lazy_policy,     n_episodes=50, seed=0)
    rand = evaluate(random_policy,   n_episodes=50, seed=0)
    scr  = evaluate(scripted_policy, n_episodes=50, seed=0)

    # Scripted solves the task. Lazy fails quickly. Random fails slowly.
    assert scr["reward_mean"] > lazy["reward_mean"] > rand["reward_mean"]
    assert scr["reward_mean"] - lazy["reward_mean"] > 1.0
    assert lazy["reward_mean"] - rand["reward_mean"] > 0.0


def test_scripted_is_efficient():
    stats = evaluate(scripted_policy, n_episodes=50, seed=0)
    assert stats["steps_mean"] <= 5     
    assert stats["match_mean"] > 0.9   