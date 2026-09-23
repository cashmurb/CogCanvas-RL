from cogcanvas.gym_wrapper import GymCanvasEnv

env = GymCanvasEnv(seed=0)
obs, info = env.reset()
print(f"reset returns: obs={type(obs).__name__}, info={type(info).__name__}")

result = env.step(0)
print(f"step returns {len(result)} items")
for i, item in enumerate(result):
    print(f"  [{i}] {type(item).__name__}")

print(f"info dict contents: {result[4]}")