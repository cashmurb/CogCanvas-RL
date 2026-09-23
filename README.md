# CogCanvas-RL 

A small reinforcement learning project where an AI agent edits a canvas step-by-step to solve a visual task by placing marks on a grid to match a target pattern.

The agent learns by trial and error, gets scored on how well it does, and gradually improves. 

This project dove into two research questions:

1. Can we score "clean" and "tidy" without the agent cheating (like drawing nothing)?
2. Does a Process Reward Model (PRM) help the agent learn faster?


| Phase | What | Result |
|---|---|---|
| 1 | The environment (an 8×8 grid) | ✓ |
| 2 | The reward (how the agent is scored) | ✓ |
| 3 | Three test policies (dumb, random, smart) | ✓ |
| 4 | TensorBoard logging | ✓ |
| 5 | Training with PPO | ✓ 90% success |
| 6 | Heuristic PRM | ✓ tested |
| 7 | Learned PRM | ✓ tested |
| 8 | Scaling to 16×16 (attempt 1) | ✗ failed |
| 9 | Scaling to 16×16 (attempt 2) and 12×12 probe | ✗ failed |

## The task

An 8×8 grid. The environment picks 3 cells at random as targets. The agent places and erases marks, trying to match the target. The episode ends when the agent calls STOP or runs out of steps.

## The reward

The agent gets a score after every action:

```
Score = 10 × correctness
      + 5 × (improvement since last step)
      + cleanliness bonus    (only if correctness > 0.9)
      − 0.01 per step
```

Two rules prevent cheating:

- **Task floor.** Correctness below 85% earns no task reward. Prevents partial credit for a messy canvas.
- **Gate.** Cleanliness only counts if correctness is above 90%. Prevents the agent from drawing nothing and scoring high on tidiness.

## Results at 8×8

Trained for 500,000 steps on CPU in about 7 minutes.

| Metric | Value |
|---|---|
| Success rate | **90%** |
| Mean reward | +14.2 |
| Mean steps | 8.0 |

The agent reliably places the 3 correct marks and stops.


## What we learned

**Finding 1: Shaping is necessary.**
Without a per-step reward for improvement, the agent collapses. It learns to STOP immediately because stopping avoids step cost. Adding `5 × (improvement)` per step makes training work.

**Finding 2: Two thresholds are needed.**
A task floor (85%) prevents partial credit for messy canvases. A gate (90%) prevents cleanliness rewards on wrong answers. Both were added after tests showed they were necessary.

**Finding 3: A heuristic PRM improves efficiency, not correctness.**
Three training runs were compared:

| Model | Success rate | Steps |
|---|---|---|
| No PRM | 90.0% | 8.0 |
| Heuristic PRM | 86.0% | 6.6 |
| Learned PRM | 82.0% | 6.4 |

The heuristic PRM finishes episodes in fewer steps but doesn't solve more. It teaches the agent to give up early when it can't win.

**Finding 4: A learned PRM that imitates the heuristic still degrades the policy.**
The learned PRM copied the heuristic to within 0.012 error. But when used for training, it produced a slightly worse policy (82% vs 86%). Small errors compound when the agent optimizes against them.

**Finding 5: The system does not scale past 8×8.**
Three attempts to train at 12×12 and 16×16 failed. The reason is reward density: at bigger grids, random placement hits a target far less often, so the reward signal is too sparse for the agent to learn from.

| Grid | Hit rate | Result |
|---|---|---|
| 8×8 | 4.7% | works |
| 12×12 | 2.1% | fails |
| 16×16 | 1.6% | fails |

This boundary is clear. Changing the action space, the network, or the hyperparameters did not help.


## Requirements
- Python 3.11+ 
- numpy, pytest, tensorboardX, torch, gymnasium, stable-baseline3 

Install:
```
pip install numpy pytest tensorboardX
pip install "stable-baselines3==2.3.0" "gymnasium==0.29.1"
```

## How to run

**Run all tests:**

```bash
python -m pytest tests/ -v
```

**Train PPO from scratch:**

```bash
python -m cogcanvas.train_ppo
```

**Train with the heuristic PRM:**

```bash
python -m cogcanvas.train_ppo --prm
```

**Train with the learned PRM:**

```bash
python -m cogcanvas.train_prm           # trains the PRM first
python -m cogcanvas.train_ppo --learned-prm
```
**Evaluate a saved model:**

```bash
python -m cogcanvas.eval_ppo runs/ppo_rung0_noprm/final_model
```

**View TensorBoard:**

```bash
tensorboard --logdir runs/
```

Open `http://localhost:6006`.

## Project Structure
```
cogcanvas/
  env.py            
  metrics.py        
  baselines.py       
  logger.py         
  gym_wrapper.py     
  prm.py             
  prm_learned.py     
  prm_data.py        
  train_ppo.py       
  train_prm.py       
  eval_ppo.py       
tests/               
runs/                
PHASE*.md            
```

## What's next

Scaling beyond 8×8 would require a different approach:

- A denser reward signal (for example, reward the agent for getting closer to a target, not just for hitting it)
- A curriculum that starts easier and gets harder
- Warm-starting the policy by imitating a scripted solution

None of these are hyperparameter changes. Each would be a new project with its own plan.

## License

MIT.
