import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
import numpy as np
import pickle
import wandb
import random
import argparse

ENABLE_WANDB = True

# --- Model Definition ---


class RegularMLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state)

    def act(self, state):
        return self.forward(state)
    
    def loss(self, state, action):
        pred_actions = self.forward(state)
        return F.mse_loss(pred_actions, action)


class GaussianMLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_head(x)
        # Clamping log_std for numerical stability
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def log_prob(self, state, action):
        """Returns the log probability of an expert action given the state."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        # Summing log_probs over action dimensions (diagonal Gaussian assumption)
        return dist.log_prob(action).sum(dim=-1)

    def act(self, state):
        """Deterministic action (argmax) for inference."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.forward(state)
        return mu
    
    def loss(self, state, action):
        log_probs = self.log_prob(state, action)
        return -log_probs.mean()


# --- Dataset Wrapper ---

class TrajectoryDataset(Dataset):
    def __init__(self, processed_data):
        states = [traj['current'] for traj in processed_data]
        actions = [traj['label'] for traj in processed_data]
        
        self.states = torch.from_numpy(np.concatenate(states, axis=0)).float()
        self.actions = torch.from_numpy(np.concatenate(actions, axis=0)).float()

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# --- Training and Eval Functions ---

def train_behavior_cloning(model, train_data, val_data, epochs=100, lr=3e-4, batch_size=64, device="cuda", save_path="checkpoints"):
    train_loader = DataLoader(
        TrajectoryDataset(train_data), 
        batch_size=batch_size, shuffle=True, num_workers=4,
    )
    val_loader = DataLoader(
        TrajectoryDataset(val_data), 
        batch_size=batch_size, shuffle=False, num_workers=4,
    )

    os.makedirs(save_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for states, actions in pbar:
            states, actions = states.to(device), actions.to(device)
            
            optimizer.zero_grad()

            loss = model.loss(states, actions)
            
            loss.backward()
            optimizer.step()
            train_loss += min(loss.item(), 100)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                val_loss += min(model.loss(states, actions).item(), 100)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Summary - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if ENABLE_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr']
            })

        # Checkpoint every 10 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            csp = os.path.join(save_path, f"{epoch}-ckpt.pt")
            torch.save(model.state_dict(), csp)
            print(f"Model at epoch {epoch} saved to {csp}")

def load_robot_policy(save_path, device="cpu"):
    with open(os.path.join(os.path.dirname(save_path), "info.pkl"), "rb") as fi:
        save_dict = pickle.load(fi)
    
    if save_dict['use_gaussian']:
        model = GaussianMLPPolicy(save_dict['current_dim'], save_dict['label_dim'], save_dict['d_model'])
    else:
        model = RegularMLPPolicy(save_dict['current_dim'], save_dict['label_dim'], save_dict['d_model'])
    
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()
    return model, save_dict

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Train Robot Transformer Policy")
    
    # Adding parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--action_low", type=float, default=-0.999, help="Minimum action value")
    parser.add_argument("--action_high", type=float, default=0.999, help="Maximum action value")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer & MLP hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--save_path", type=str, default="policy_checkpoint.pt", help="Path to save the model")
    parser.add_argument("--dataset_path", type=str, default="N/A", help="Path to load the dataset")
    parser.add_argument("--use_gaussian", action="store_true", default=False, help="Use Gaussian policy.")
    
    args = parser.parse_args()

    # Accessing the parameters
    LR = args.lr
    EPOCHS = args.epochs
    ACTION_LOW = args.action_low
    ACTION_HIGH = args.action_high
    BATCH_SIZE = args.batch_size

    GAUSSIAN = args.use_gaussian
    D_MODEL = args.d_model
    NUM_LAYERS = args.num_layers
    DROPOUT = args.dropout

    save_path = args.save_path
    
    CONTEXT_DIM = 45 + 7
    CURRENT_DIM = 45 * 5
    LABEL_DIM = 7

    if ENABLE_WANDB:
        wandb.init(project="robot-mlp-bc", config=vars(args))
    
    DATASET_PATH = args.dataset_path

    trajs = []
    try:
        with open(DATASET_PATH, "rb") as fi:
            trajs += pickle.load(fi)
    except FileNotFoundError:
        print("Data file not found.")
        return
    print("Loaded dataset.")
    
    
    all_labels = np.concatenate([traj['actions_expert'] for traj in trajs], axis=0)
    label_means = all_labels.mean(axis=0)
    label_stds = all_labels.std(axis=0)
    print(f"Label means = {label_means.tolist()}")
    print(f"Label stds = {label_stds.tolist()}")

    save_dict = {
        'dataset_origin': os.path.abspath(DATASET_PATH),
        'label_stds': label_stds.tolist(),
        'context_dim': CONTEXT_DIM,
        'current_dim': CURRENT_DIM,
        'label_dim': LABEL_DIM,
        'use_gaussian': GAUSSIAN,
        'd_model': D_MODEL,
    }
    if os.path.exists(os.path.join(os.path.dirname(DATASET_PATH), "info.pkl")):
        with open(os.path.join(os.path.dirname(DATASET_PATH), "info.pkl"), "rb") as fi:
            load_dict = pickle.load(fi)
        save_dict |= {
            'use_noise_scales': load_dict['use_general_scales'],
            'sys_noise_scale': load_dict['sys_noise_scale'],
            'rand_noise_scale': load_dict['rand_noise_scale'],
            'obs_insertive_noise_scale': load_dict['obs_insertive_noise_scale'],
            'obs_receptive_noise_scale': load_dict['obs_receptive_noise_scale'],
        }
    else:
        save_dict |= {
            'use_noise_scales': True,
            'sys_noise_scale': 0,
            'rand_noise_scale': 0,
            'obs_insertive_noise_scale': float(os.path.basename(DATASET_PATH)[:-4].split('-')[-1]),
            'obs_receptive_noise_scale': float(os.path.basename(DATASET_PATH)[:-4].split('-')[-2]),
        }
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "info.pkl"), "wb") as fi:
        pickle.dump(save_dict, fi)

    processed_data = []
    for traj in trajs:
        if traj['rewards'].ndim == 1:
            traj['rewards'] = traj['rewards'][:, None]
        
        processed_data.append({
            'current': traj['obs']['policy'],
            'label': (traj['actions_expert'] - traj['actions']) / label_stds,
        })
    assert processed_data[0]['current'].shape[-1] == CURRENT_DIM
    assert processed_data[0]['label'].shape[-1] == LABEL_DIM

    random.shuffle(processed_data)
    split = int(len(processed_data) * 0.8)
    train_data = processed_data[:split]
    val_data = processed_data[split:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if GAUSSIAN:
        model = GaussianMLPPolicy(CURRENT_DIM, LABEL_DIM, D_MODEL)
    else:
        model = RegularMLPPolicy(CURRENT_DIM, LABEL_DIM, D_MODEL)
    model.to(device)
    if ENABLE_WANDB:
        wandb.watch(model)

    try:
        train_behavior_cloning(model, train_data, val_data, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, device=device, save_path=save_path)
    finally:
        if ENABLE_WANDB:
            wandb.finish()

if __name__ == '__main__':
    main()
