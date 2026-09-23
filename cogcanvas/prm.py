import numpy as np 
from cogcanvas.metrics import f1_score, clutter, abstraction

class HeuristicPRM:
    """
    Scores a state in [0,1]. Higher = more promising.
    Combines F1 score (task correctness), clutter(visual tidiness), and abstraction (economy of marks) 
    """

    def __init__(self, cap=6):
        self.cap = cap

    def score(self, obs):
        canvas = obs[0].astype(np.int8)
        target = obs[1].astype(np.int8)

        f1 = f1_score(canvas, target)
        c = clutter(canvas, self.cap)
        a = abstraction(canvas, self.cap)

        return 0.8 * f1 + 0.1 * (1 - c) + 0.1 * a
        