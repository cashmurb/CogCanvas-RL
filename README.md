# CogCanvas-RL 

A local prototype of an AI agent that edits a canvas step-by-step to turn a cluttered input into a clean structural abstraction. This project dove into two research questions:

1. Can we measure "visual clutter" in a way that resists reward hacking?
2. Does a Process Reward Model (PRM) help a visual agent learn faster?

Everything else including the environment, the reward, the training, the logging is scaffolding to answer those. 

## What's build
| Phase | What | Status |
|---|---|---|
| 1 | Environment (`CanvasEnv`) | ✓ |
| 2 | Reward: F1 + clutter + abstraction + gate | ✓ |
| 3 | Baselines: lazy, random, scripted | ✓ |
| 4 | TensorBoard logging | ✓ |
| 5 | PPO training, Rung 0 solved (90% success) | ✓ |
| 6 | Heuristic PRM | ✓ |
| 7 | Learned PRM | ✓ |

## Key Findings
- **Shaping matters.** Sparse terminal-only reward causes policy collapse. Adding `5.0 x F1` per step makes Rung 0 trainable in ~7 minutes on CPU. 
- **Two thresholds are needed.** A task floor (`F1 > 0.85`) prevents partial credit for messy canvases. A gate (`F1 > 0.9`) prevents cleanliness bonuses on wrong answers. 
- **The heuristic PRM improves efficiency, not correctness.** PRM-trained policies finish episodes in ~18% fewer steps but do not solve more often. 
- **A learned PRM that imitates the heuristic is not enough.** Near-perfect imitation (RMSE 0.012) still degrades the downstream policy. Small errors compound. 

## Requirements
- Python 3.11 + 
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
python -m cogcanvas.eval_ppo runs/ppo_rung0/final_model
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

## The task (Rung 0)

An 8x8 grid. The environment picks 3 target cells at random. The agent places and erases marks to match the target. The episode ends when the agent calls STOP or reaches 12 steps. 

**Reward:**

```
R = 10 × F1 (if F1 > 0.85)
  + 5 × (F1_now − F1_prev) (every step)
  + gate × (clutter_bonus + abstraction_bonus)
  − 0.01 × step_cost
```

where `gate = 1` only if `F1 > 0.9`

## Results at Rung 0 
| Model | Success rate | Mean steps |
|---|---|---|
| No PRM | 90.0% | 8.0 |
| Heuristic PRM | 86.0% | 6.6 |
| Learned PRM | 82.0% | 6.4 |

## License

MIT.