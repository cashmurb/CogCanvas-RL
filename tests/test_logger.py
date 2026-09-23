import os
import tempfile
import shutil
from cogcanvas.logger import TBLogger


def test_logger_creates_files():
    tmp = tempfile.mkdtemp()
    try:
        logger = TBLogger(log_dir=tmp, window=5)
        for ep in range(10):
            logger.log_episode(
                episode=ep,
                total_reward=-0.2,
                steps=24,
                info={"match": 0.0, "clutter": 0.0, "abstraction": 1.0,
                      "task": 0.0, "clutter_bonus": 0.0,
                      "abstraction_bonus": 0.0, "step_cost": 0.01},
            )
        logger.close()

        files = os.listdir(tmp)
        assert any(f.startswith("events.out") for f in files)
    finally:
        shutil.rmtree(tmp)


def test_logger_rolling_mean():
    tmp = tempfile.mkdtemp()
    try:
        logger = TBLogger(log_dir=tmp, window=5)
        for ep in range(10):
            logger.log_episode(ep, total_reward=float(ep), steps=1, info={})
        logger.close()

        assert len(logger.reward_history) == 10
        assert abs(sum(logger.reward_history[-5:]) / 5 - 7.0) < 1e-9
    finally:
        shutil.rmtree(tmp)