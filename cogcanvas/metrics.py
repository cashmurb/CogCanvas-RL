import numpy as np

def f1_score(canvas, target):
    pred = (canvas.flatten() == 1)
    true = (target.flatten() == 1)

    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())

    if tp == 0:
        return 0.0 
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0 
    return 2 * (precision * recall) / (precision + recall)


def edge_density(canvas):
    h = (canvas[:,1:] != canvas[:, :-1]).mean() if canvas.shape[1] > 1 else 0.0
    v = (canvas[1:, :] != canvas[:-1, :]).mean() if canvas.shape[0] > 1 else 0.0
    return float((h+v)/2)

def primitive_density(canvas,cap):
    marks = int((canvas == 1).sum())
    return float(min(marks / cap, 1.0))

def clutter(canvas, cap, w_edge=0.5, w_prim=0.5):
    return w_edge * edge_density(canvas) + w_prim * primitive_density(canvas, cap)

def abstraction(canvas, cap):
    marks = int((canvas == 1).sum())
    return float(max(0.0, 1.0 - marks / cap))