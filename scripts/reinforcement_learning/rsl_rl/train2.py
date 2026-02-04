import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import wandb
import random
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


ENABLE_WANDB = True

def save_histogram(x, filename, bins=100):
    # convert to numpy
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    x = x.reshape(-1)

    plt.figure()
    plt.hist(x, bins=bins)
    plt.tight_layout()
    plt.savefig(filename)
    print(filename)
    plt.close()

def model_has_nan_or_inf(model):
    for name, param in model.named_parameters():
        if param is None:
            continue
        if not torch.isfinite(param).all():
            print(f"Non-finite values in parameter: {name}")
            return True
    return False

# --- Model Definition ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class RobotTransformerPolicy(nn.Module):
    def __init__(self, context_dim, current_dim, label_dim, nhead=8, num_layers=4, d_model=512, dropout=0.1):
        super().__init__()
        self.context_proj = nn.Linear(context_dim, d_model)
        self.ctx_norm = nn.LayerNorm(d_model)
        self.current_proj = nn.Linear(current_dim, d_model)
        self.curr_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, label_dim)
        )

    def forward(self, context, current, padding_mask=None):
        ctx_emb = self.ctx_norm(self.context_proj(context))
        ctx_emb = self.pos_encoder(ctx_emb)
        
        # padding_mask: (Batch, Seq_Len)
        ctx_out = self.transformer(ctx_emb, src_key_padding_mask=padding_mask)
        
        # IMPORTANT: When pooling, we must ignore the padded values
        if padding_mask is not None:
            # Mask the output to 0 before averaging
            # ~padding_mask.unsqueeze(-1) flips True/False and adds a dim
            ctx_out = ctx_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            # Calculate actual lengths to get a true mean
            lengths = (~padding_mask).sum(dim=1, keepdim=True)
            ctx_agg = ctx_out.sum(dim=1) / lengths
        else:
            ctx_agg = torch.mean(ctx_out, dim=1)
            
        curr_emb = self.curr_norm(self.current_proj(current))
        combined = torch.cat([ctx_agg, curr_emb], dim=-1)
        return self.head(combined)

class IndependentTrajectoryDataset(Dataset):
    def __init__(
            self,
            data,
            train_mode,
            closest_neighbors_radius: float = 0,
        ):
        """
        data: List of dicts containing 'context', 'current', 'label', and 'choosable'
        """
        self.data = data

        # 2. Flatten all states from ALL trajectories for 'current' sampling
        self.all_currents = []
        for traj in data:
            # We take the current states from all trajectories, even unchoosable ones
            self.all_currents.append(traj['current'])
        
        self.all_currents = np.concatenate(self.all_currents, axis=0)

        self.train_mode = train_mode
        if train_mode == "closest-neighbors":
            assert closest_neighbors_radius > 0
            self.choosable_trajs = []
            self.closest_neighbors_radius = closest_neighbors_radius
            self.all_receptive_noises = np.stack([traj['obs_receptive_noise'] for traj in data], axis=0)
            self.valid_seconds = []
            for i, traj in tqdm(enumerate(data)):
                if not traj.get('choosable', False):
                    continue
                cur_distances = np.linalg.norm(self.all_receptive_noises - traj['obs_receptive_noise'], axis=-1)
                if (((cur_distances <= closest_neighbors_radius) & (cur_distances > 0)).sum() == 0):
                    continue
                cur_seconds = np.where((cur_distances <= closest_neighbors_radius) & (cur_distances > 0))[0]
                self.choosable_trajs.append(traj)
                self.valid_seconds.append(cur_seconds)
                if i < 20:
                    print(self.valid_seconds[-1].shape)
        elif train_mode == "single-traj":
            self.choosable_trajs = [traj for traj in data if traj.get('choosable', False)]
        else:
            raise NotImplementedError(train_mode)

    def __len__(self):
        # The epoch length is defined by how many choosable demonstration sequences we have
        return len(self.choosable_trajs)

    def __getitem__(self, idx):
        # Get the context and label from a "choosable" trajectory
        traj = self.choosable_trajs[idx]

        if self.train_mode == "closest-neighbors":
            context = torch.tensor(traj['context'], dtype=torch.float32)
            second_traj = self.data[np.random.choice(self.valid_seconds[idx])]
            st = random.randint(0, second_traj['current'].shape[0] - 1)
            current = torch.tensor(second_traj['current'][st], dtype=torch.float32)
            label = torch.tensor(second_traj['label'][st], dtype=torch.float32)
        elif self.train_mode == "single-traj":
            T = traj['context'].shape[0]
            assert T > 6, f"{T}"
            zt = random.randint(6, T - 1)
            st = random.randint(zt, T - 1)
            context = torch.tensor(traj['context'][:zt], dtype=torch.float32)
            current = torch.tensor(traj['current'][st], dtype=torch.float32)
            label = torch.tensor(traj['label'][st], dtype=torch.float32)
        
        return context, current, label

def collate_fn(batch):
    """
    Custom collator to pad trajectories of different lengths.
    """
    contexts, currents, labels = zip(*batch)
    
    # Pad sequences to the max length in this specific batch
    # padded_contexts shape: (Batch, Max_T, Context_Dim)
    padded_contexts = torch.nn.utils.rnn.pad_sequence(contexts, batch_first=True)
    
    # Create a mask: True for padded positions, False for real data
    # This is for PyTorch's src_key_padding_mask
    padding_mask = torch.zeros(padded_contexts.shape[0], padded_contexts.shape[1], dtype=torch.bool)
    for i, ctx in enumerate(contexts):
        padding_mask[i, len(ctx):] = True
        
    currents = torch.stack(currents)
    labels = torch.stack(labels)
    
    return padded_contexts, currents, labels, padding_mask

def train_behavior_cloning(
        model,
        train_data,
        val_data,
        epochs=100,
        lr=1e-4,
        batch_size=64,
        device="cuda",
        save_path=None,
        train_mode: str = "single-traj",
        closest_neighbors_radius: float = 0.001,
        warm_start: int = 0,
    ):
    train_loader = DataLoader(
        IndependentTrajectoryDataset(
            train_data,
            train_mode=train_mode,
            closest_neighbors_radius=closest_neighbors_radius,
        ),
        batch_size=batch_size, shuffle=True, num_workers=4, 
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        IndependentTrajectoryDataset(
            val_data,
            train_mode=train_mode,
            closest_neighbors_radius=closest_neighbors_radius,
        ),
        batch_size=batch_size, shuffle=False, num_workers=4, 
        collate_fn=collate_fn, pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    # Learning rate scheduler for better convergence
    if warm_start <= 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm_start),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs - warm_start),
            ],
            milestones=[warm_start],
        )

    fixed_epochs = []
    best_loss = 100000
    best_loss_epoch = -1
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for context, current, label, padding_mask in pbar:
            context, current, label = context.to(device), current.to(device), label.to(device)
            padding_mask = padding_mask.to(device)
            
            optimizer.zero_grad()
            pred = model(context, current, padding_mask)
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for context, current, label, padding_mask in val_loader:
                context, current, label = context.to(device), current.to(device), label.to(device)
                padding_mask = padding_mask.to(device)
                pred = model(context, current, padding_mask)
                vloss = criterion(pred, label)
                val_loss += vloss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Summary - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if ENABLE_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr']
            })
        
        if epoch % 50 == 49 and save_path is not None:
            csp = os.path.join(save_path, f"{epoch}-ckpt.pt")
            torch.save(model.state_dict(), csp)
            print(f"Model at epoch {epoch} saved to {csp}")
            fixed_epochs.append(epoch)
        
        if epoch > 40 and avg_val_loss < best_loss and save_path is not None and epoch not in fixed_epochs:
            best_loss = avg_val_loss
            if best_loss_epoch not in fixed_epochs:
                csp = os.path.join(save_path, f"{best_loss_epoch}-ckpt.pt")
                if os.path.exists(csp):
                    os.unlink(csp)
                    print(f"Model at epoch {best_loss_epoch} removed.")
            best_loss_epoch = epoch
            csp = os.path.join(save_path, f"{epoch}-ckpt.pt")
            torch.save(model.state_dict(), csp)
            print(f"Best model at epoch {epoch} saved to {csp}")

    
    if save_path is not None:
        csp = os.path.join(save_path, f"{epochs}-ckpt.pt")
        torch.save(model.state_dict(), csp)
        print(f"Model at epoch {epochs} saved to {csp}")

def load_robot_policy(save_path, device="cpu"):
    with open(os.path.join(os.path.dirname(save_path), "info.pkl"), "rb") as fi:
        save_dict = pickle.load(fi)
    save_dict = {
        'current_means': np.zeros((save_dict['current_dim'],)),
        'current_stds': np.ones((save_dict['current_dim'],)),
        'context_means': np.zeros((save_dict['context_dim'],)),
        'context_stds': np.ones((save_dict['context_dim'],)),
        'label_means': np.zeros((save_dict['label_dim'],)),
        'label_stds': np.ones((save_dict['label_dim'],)),
        'train_expert': False,
    } | save_dict
    model = RobotTransformerPolicy(
        save_dict['context_dim'],
        save_dict['current_dim'],
        save_dict['label_dim'],
        num_layers=save_dict['num_layers'],
        d_model=save_dict['d_model'],
        dropout=save_dict['dropout'],
    )
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
    parser.add_argument("--train_mode", type=str, default="single-traj", help="Options: single-traj, closest-neighbors.")
    parser.add_argument("--closest_neighbors_radius", type=float, default=0.001, help="If train_mode is closest-neighbors.")
    parser.add_argument("--warm_start", type=int, default=0, help="Number of warm start epochs.")
    parser.add_argument("--train_percent", type=float, default=0.8, help="Percentage of data used for train.")
    parser.add_argument("--train_expert", action="store_true", default=False, help="Whether we're training an expert or a residual.")
    
    args = parser.parse_args()

    # Accessing the parameters
    LR = args.lr
    EPOCHS = args.epochs
    ACTION_LOW = args.action_low
    ACTION_HIGH = args.action_high
    BATCH_SIZE = args.batch_size

    D_MODEL = args.d_model
    NUM_LAYERS = args.num_layers
    DROPOUT = args.dropout

    save_path = args.save_path
    
    CONTEXT_DIM = 45 + 7
    CURRENT_DIM = 45
    LABEL_DIM = 7

    if ENABLE_WANDB:
        wandb.init(project="robot-transformer-bc-deterministic-normalized-labels", config=vars(args))
    
    DATASET_PATH = args.dataset_path

    trajs = []
    try:
        with open(DATASET_PATH, "rb") as fi:
            trajs += pickle.load(fi)
    except FileNotFoundError:
        print("Data file not found.")
        return
    print("Loaded dataset.")

    processed_data = []
    for traj in trajs:
        if traj['rewards'].ndim == 1:
            traj['rewards'] = traj['rewards'][:, None]
        
        processed_traj = {
            'context': np.concatenate([traj['obs']['policy2'], traj['actions']], axis=1),
            'current': traj['obs']['policy2'],
            'label': traj['actions_expert'] - traj['actions'],
            'choosable': traj['obs']['policy2'].shape[0] > 6,
            'obs_receptive_noise': traj['obs_receptive_noise'],
            # 'choosable': not np.any(traj['rewards'] > 0.11),
        }
        if args.train_expert:
            processed_traj['context'] *= 0
            processed_traj['label'] = traj['actions_expert']
        
        processed_data.append(processed_traj)
    assert processed_data[0]['context'].shape[-1] == CONTEXT_DIM
    assert processed_data[0]['current'].shape[-1] == CURRENT_DIM
    assert processed_data[0]['label'].shape[-1] == LABEL_DIM

    # Current normalization
    all_currents = np.concatenate([traj['current'] for traj in processed_data], axis=0)
    current_means = all_currents.mean(axis=0)
    current_stds = all_currents.std(axis=0)
    all_contexts = np.concatenate([traj['context'] for traj in processed_data], axis=0)
    context_means = all_contexts.mean(axis=0)
    context_stds = all_contexts.std(axis=0) + 1e-9
    all_labels = np.concatenate([traj['label'] for traj in processed_data], axis=0)
    label_means = all_labels.mean(axis=0)
    label_stds = all_labels.std(axis=0)
    for traj in processed_data:
        traj['current'] = (traj['current'] - current_means) / current_stds
        traj['context'] = (traj['context'] - context_means) / context_stds
        traj['label'] = (traj['label'] - label_means) / label_stds

    save_dict = {
        'dataset_origin': os.path.abspath(DATASET_PATH),
        'save_path': save_path,
        'current_means': current_means,
        'current_stds': current_stds,
        'context_means': context_means,
        'context_stds': context_stds,
        'label_means': label_means,
        'label_stds': label_stds,
        'context_dim': CONTEXT_DIM,
        'current_dim': CURRENT_DIM,
        'label_dim': LABEL_DIM,
        'd_model': D_MODEL,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'train_mode': args.train_mode,
        'closest_neighbors_radius': args.closest_neighbors_radius,
        'warm_start': args.warm_start,
        'train_percent': args.train_percent,
        'train_expert': args.train_expert,
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

    viz_path = os.path.join(save_path, "viz")
    os.makedirs(viz_path, exist_ok=True)
    all_labels_viz = np.concatenate([traj['label'] for traj in processed_data], axis=0)
    for i in range(LABEL_DIM):
        save_histogram(all_labels_viz[:, i], os.path.join(viz_path, f"label_{i}.png"))

    num_choosable = sum(1 for d in processed_data if d['choosable'])
    print(f"Total Trajectories: {len(processed_data)}")
    print(f"Choosable Trajectories: {num_choosable}")
    
    if num_choosable == 0:
        print("Error: No choosable trajectories found. Check reward thresholds.")
        return

    random.shuffle(processed_data)
    split = int(len(processed_data) * args.train_percent)
    train_data = processed_data[:split]
    val_data = processed_data[split:]
    print(f"Train percent: {args.train_percent} !")

    # Final safeguard: ensure both splits have at least one choosable traj
    if not any(d['choosable'] for d in val_data):
        print("Warning: Validation set has no choosable trajectories. Re-shuffling...")
        # In a real scenario, you might want a Stratified Split here

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RobotTransformerPolicy(CONTEXT_DIM, CURRENT_DIM, LABEL_DIM, num_layers=NUM_LAYERS, d_model=D_MODEL, dropout=DROPOUT)
    model.to(device)
    if ENABLE_WANDB:
        wandb.watch(model)

    try:
        train_behavior_cloning(
            model,
            train_data,
            val_data,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            device=device,
            save_path=save_path,
            train_mode=args.train_mode,
            closest_neighbors_radius=args.closest_neighbors_radius,
            warm_start=args.warm_start,
        )
    finally:
        if ENABLE_WANDB:
            wandb.finish()

if __name__ == '__main__':
    main()
