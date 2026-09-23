import numpy as np
from cogcanvas.metrics import f1_score, clutter, abstraction

class CanvasEnv:
    """
    8x8 grid where agent places and erases marks.
    Goal: match the target
    Action Space (size 2*N + 1, where N = size*size):
        [0, N) -> place a mark at cell `action`
        [N, 2N) -> erase a mark at cell `action - N`
        [2N] -> stop
    Cell index -> (y, x) via divmod(index, size)
    """

    def __init__(self, size=8, n_targets=3, max_steps=24, seed=None, w_task=10.0, w_clutter=0.5, w_abs=0.5, w_step=0.01, gate_threshold=0.9, w_shaping=5.0):
        self.size = size 
        self.n_targets = n_targets 
        self.max_steps = max_steps
        self.n_cells = size * size
        self.n_actions = 2 * self.n_cells + 1
        self.rng = np.random.default_rng(seed)

        # reward settings
        self.w_task = w_task
        self.w_clutter = w_clutter
        self.w_abs = w_abs
        self.w_step = w_step
        self.gate_threshold = gate_threshold
        self.cap = 2 * n_targets
        self.w_shaping = w_shaping
        

        # state
        self.canvas = None
        self.target = None
        self.t = 0 
        self.done = False
        self.prev_f1 = 0.0

    # core API 
    def reset(self):
        idx = self.rng.choice(self.n_cells, size=self.n_targets, replace=False)
        self.target = np.zeros((self.size, self.size), dtype=np.int8)
        for i in idx:
            y, x = divmod(int(i), self.size)
            self.target[y, x] = 1

        self.canvas = np.zeros((self.size, self.size), dtype=np.int8)
        self.t = 0
        self.done = False
        self.prev_f1 = 0.0
        return self._obs()

    def step(self, action):
        if self.done:
            raise ValueError("step() called after episode ended")

        self.t += 1
        n = self.n_cells

        if action == 2*n:
            self.done = True
        elif action < n:
            y, x = divmod(action, self.size)
            self.canvas[y, x] = 1
        else:
            y, x = divmod(action - n, self.size)
            self.canvas[y, x] = 0
        
        if self.t >= self.max_steps:
            self.done = True
        
        # compute reward
        step_cost = self.w_step
        task_r = 0.0
        clutter_r = 0.0
        abstraction_r = 0.0

        # F1-delta shaping: fires every step, rewards progress
        current_f1 = f1_score(self.canvas, self.target)
        shaping_r = self.w_shaping * (current_f1 - self.prev_f1)
        self.prev_f1 = current_f1

        if self.done:
            if current_f1 > 0.85:
                task_r = self.w_task * current_f1
            if current_f1 > self.gate_threshold:
                c = clutter(self.canvas, self.cap)
                a = abstraction(self.canvas, self.cap)
                clutter_r = self.w_clutter * (1 - c)
                abstraction_r = self.w_abs * a

        reward = task_r + shaping_r + clutter_r + abstraction_r - step_cost
        return self._obs(), reward, self.done, self._info()
    
    # helpers 
    def _obs(self):
        return np.stack([self.canvas, self.target]).astype(np.float32)
    
    def _info(self):
        match = float((self.canvas == self.target).mean())
        f1 = f1_score(self.canvas, self.target)
        c = clutter(self.canvas, self.cap)
        a = abstraction(self.canvas, self.cap)

        # reward earned this step 
        task_r = self.w_task * f1 if (self.done and f1 > 0.85) else 0.0
        gate = 1.0 if (self.done and f1 > self.gate_threshold) else 0.0 
        clutter_r = self.w_clutter * (1 - c) * gate 
        abstraction_r = self.w_abs * a * gate 

        return {
            "match": f1, 
            "task": task_r,
            "clutter": c,
            "abstraction": a,
            "clutter_bonus": clutter_r,
            "abstraction_bonus": abstraction_r,
            "step_cost": self.w_step,
            "steps": self.t,
        }
            
    
    def render(self, scale=16):
        """
        Return an RGB image of the current state. For visualisation purposes only. 
        """

        H, W = self.size, self.size
        img = np.full((H, W, 3), 255, dtype=np.uint8)
        img[self.target == 1] = [210, 210, 210] # target: light gray
        img[self.canvas == 1] = [0, 0, 0] # agent marks: black

        return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))