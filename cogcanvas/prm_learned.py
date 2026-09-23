import numpy as np
import torch
import torch.nn as nn

class LearnedPRMNet(nn.Module):
    """ Small CNN: (2, 8, 8) -> scalar. """

    def __init__(self, in_channels=2, size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * size * size, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class LearnedPRM:
    """ Wrapper matching the HeuristicPRM interface: .score(obs) -> float. """

    def __init__(self, model_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model = LearnedPRMNet().to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def score(self, obs):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs)).unsqueeze(0).float().to(self.device)
            return float(self.model(x).item())

    def save(self, path):
        torch.save(self.model.state_dict(), path)