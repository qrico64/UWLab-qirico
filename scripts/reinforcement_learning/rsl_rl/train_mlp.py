import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import wandb
import random

ENABLE_WANDB = True

# --- Model Definition ---

class MarkovianPolicy(nn.Module):
    """
    A standard MLP policy for Markovian decision processes.
    Maps current state (current_dim) -> action (label_dim).
    """
    def __init__(self, current_dim, label_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, label_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Dataset Wrapper ---

class TrajectoryDataset(Dataset):
    def __init__(self, processed_data):
        # We flatten all trajectories into a single pool of (state, action) pairs
        self.states = []
        self.actions = []
        for traj in processed_data:
            self.states.append(traj['current'])
            self.actions.append(traj['label'])
        
        self.states = torch.from_numpy(np.concatenate(self.states, axis=0)).float()
        self.actions = torch.from_numpy(np.concatenate(self.actions, axis=0)).float()

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# --- Training and Eval Functions ---

def train_behavior_cloning(model, train_loader, val_loader, epochs=100, lr=1e-4, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            
            optimizer.zero_grad()
            preds = model(states)
            loss = criterion(preds, actions)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                preds = model(states)
                val_loss += criterion(preds, actions).item()
        
        avg_train = epoch_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")
        if ENABLE_WANDB:
            wandb.log({"train_mse": avg_train, "val_mse": avg_val, "lr": scheduler.get_last_lr()[0]})

        if epoch % 50 == 49:
            torch.save(model.state_dict(), f"models/markovian_policy_3_s{epoch + 1}.pth")

# --- Main ---

def main():
    # Params
    LR = 3e-4 # Slightly higher for MLP
    EPOCHS = 300
    BATCH_SIZE = 256
    CURRENT_DIM = 45 * 5
    LABEL_DIM = 7
    HIDDEN_DIM = 512 * 2
    ACTION_LOW = -0.999
    ACTION_HIGH = 0.999

    if ENABLE_WANDB:
        wandb.init(
            project="robot-markov-bc",
            config={"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE},
            name="255-to-7 5layer x1024"
        )

    # Data Loading
    trajs = []
    try:
        with open("cut-trajectories_ynnn-True-0.0-0.0-10000.pkl", "rb") as fi:
            trajs += pickle.load(fi)
    except FileNotFoundError:
        print("Data file not found.")
        return
    
    # Action Normalization (Scaling logic preserved)
    action_means = np.concatenate([traj['actions'] for traj in trajs], axis=0)
    action_high = np.array([8.54, 7.43, 6.33, 16.72, 30.75, 8.65, 15.46], dtype=np.float32)
    action_low = np.array([-7.84, -10.23, -7.54, -25.27, -35.54, -6.72, -16.17], dtype=np.float32)
    print(f"Action high = {action_high.tolist()}")
    print(f"Action low = {action_low.tolist()}")
    assert np.all(np.quantile(action_means, 0.8, axis=0) < action_high) and np.all(np.quantile(action_means, 0.2, axis=0) > action_low)

    processed_data = []
    for traj in trajs:
        # Scale actions to [ACTION_LOW, ACTION_HIGH]
        norm_actions = np.clip(
            (traj['actions'] - action_low) / (action_high - action_low) * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW, 
            ACTION_LOW, ACTION_HIGH
        )
        processed_data.append({
            'current': traj['obs']['policy'],
            'label': norm_actions,
        })
    
    # Check for NaNs
    for i, traj in enumerate(processed_data):
        if np.isnan(traj['current']).any() or np.isnan(traj['label']).any():
            print(f"NaN found in trajectory {i}")
        if np.isinf(traj['current']).any() or np.isinf(traj['label']).any():
            print(f"Inf found in trajectory {i}")

    # Train/Val Split
    random.shuffle(processed_data)
    split = int(len(processed_data) * 0.8)
    
    train_set = TrajectoryDataset(processed_data[:split])
    val_set = TrajectoryDataset(processed_data[split:])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MarkovianPolicy(CURRENT_DIM, LABEL_DIM, hidden_dim=HIDDEN_DIM)
    model.to(device)
    if ENABLE_WANDB:
        wandb.watch(model)

    try:
        train_behavior_cloning(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)
    finally:
        if ENABLE_WANDB:
            wandb.finish()

if __name__ == '__main__':
    main()
