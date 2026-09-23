import numpy as np
import os
import tempfile
import torch
from cogcanvas.prm_learned import LearnedPRM, LearnedPRMNet
from cogcanvas.prm import HeuristicPRM


def _obs(canvas, target):
    return np.stack([canvas, target]).astype(np.float32)


def test_net_forward_shape():
    net = LearnedPRMNet()
    x = torch.randn(4, 2, 8, 8)
    out = net(x)
    assert out.shape == (4,)


def test_wrapper_score_is_scalar():
    prm = LearnedPRM()
    obs = _obs(np.zeros((8, 8), np.int8), np.zeros((8, 8), np.int8))
    s = prm.score(obs)
    assert isinstance(s, float)


def test_wrapper_save_and_load():
    prm1 = LearnedPRM()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        prm1.save(path)
        prm2 = LearnedPRM(model_path=path)

        obs = _obs(np.zeros((8, 8), np.int8), np.zeros((8, 8), np.int8))
        s1 = prm1.score(obs)
        s2 = prm2.score(obs)
        assert abs(s1 - s2) < 1e-5



def test_trained_prm_approximates_heuristic():
    from cogcanvas.prm_data import generate_dataset
    from cogcanvas.prm_learned import LearnedPRMNet
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X, y = generate_dataset(n_episodes=50, seed=0)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = LearnedPRMNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(10):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy()
    mae = float(np.abs(pred - y).mean())
    assert mae < 0.15, f"MAE={mae:.3f} (threshold 0.15)"

def test_learned_prm_scores_blank_below_perfect():
    """After training on mixed data, PRM should rank perfect > blank."""
    from cogcanvas.prm_data import generate_dataset
    from cogcanvas.prm_learned import LearnedPRMNet
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X, y = generate_dataset(n_episodes=100, seed=0)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = LearnedPRMNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(15):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    target = np.zeros((8, 8), np.int8)
    target[2, 3] = target[5, 1] = target[6, 6] = 1

    blank = _obs(np.zeros((8, 8), np.int8), target)
    perfect = _obs(target.copy(), target)

    with torch.no_grad():
        s_blank = float(model(torch.from_numpy(blank).unsqueeze(0)).item())
        s_perfect = float(model(torch.from_numpy(perfect).unsqueeze(0)).item())

    assert s_perfect > s_blank, f"perfect={s_perfect:.3f} blank={s_blank:.3f}"