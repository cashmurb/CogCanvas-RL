import os 
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from cogcanvas.prm_data import generate_dataset, split_dataset
from cogcanvas.prm_learned import LearnedPRMNet

def train_prm(n_episodes=500, epochs=20, batch_size=64, lr=1e-3, save_path="runs/prm/learned_prm.pt", seed=0):
    print("Generating data...")
    X, y = generate_dataset(n_episodes=n_episodes, seed=seed)
    print(f"{len(X)} samples.")

    X_tr, y_tr, X_val, y_val = split_dataset(X, y, val_frac=0.1, seed=seed)
    print(f"train: {len(X_tr)}, val: {len(X_val)}")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = LearnedPRMNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, in val_loader:
                val_losses.append(loss_fn(model(xb), yb).item())
        
        print(f"epoch {epoch+1:2d} train_mse={np.mean(train_losses):.5f}"
              f"val_mse={np.mean(val_losses):.5f}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    train_prm()
    