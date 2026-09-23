from cogcanvas.env import CanvasEnv
e = CanvasEnv(seed=0); e.reset()
for y in range(e.size):
    for x in range(e.size):
        if e.target[y, x] == 1:
            _, r, _, _ = e.step(y * e.size + x)
            print(f"Reward after hitting first target: {r:+.3f}")
            break
    else:
        continue
    break