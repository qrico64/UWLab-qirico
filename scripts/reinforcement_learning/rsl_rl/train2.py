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
import cur_utils
from train_lib import RobotTransformerPolicy


ENABLE_WANDB = True

# --- Model Definition ---

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
                    if i < 20: print("(skipped due to unchoosable)")
                    continue
                cur_distances = np.linalg.norm(self.all_receptive_noises - traj['obs_receptive_noise'], axis=-1)
                if (((cur_distances <= closest_neighbors_radius) & (cur_distances > 0)).sum() == 0):
                    if i < 20: print("(skipped due to no neighbors)")
                    continue
                cur_seconds = np.where((cur_distances <= closest_neighbors_radius) & (cur_distances > 0))[0]
                self.choosable_trajs.append(traj)
                self.valid_seconds.append(cur_seconds)
                if i < 20:
                    print(self.valid_seconds[-1].shape)
        elif train_mode == "single-traj":
            self.choosable_trajs = [traj for traj in data if traj.get('choosable', False)]
        elif train_mode == "autoregressive":
            self.choosable_trajs = [traj for traj in data if traj.get('choosable', False)]
        elif train_mode == "full-traj":
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
        elif self.train_mode == "autoregressive":
            T = traj['context'].shape[0]
            assert T > 6, f"{T}"
            zt = random.randint(1, T - 1)
            context = torch.tensor(traj['context'][:zt], dtype=torch.float32)
            current = torch.tensor(traj['current'][zt], dtype=torch.float32)
            label = torch.tensor(traj['label'][zt], dtype=torch.float32)
        elif self.train_mode == "full-traj":
            T = traj['current'].shape[0]
            assert T > 6, f"{T}"
            zt = random.randint(0, T - 1)
            context = torch.tensor(traj['context'], dtype=torch.float32)
            current = torch.tensor(traj['current'][zt], dtype=torch.float32)
            label = torch.tensor(traj['label'][zt], dtype=torch.float32)
        else:
            raise NotImplementedError(self.train_mode)
        
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

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Train Robot Transformer Policy")
    
    # Adding parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer & MLP hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--save_path", type=str, default="policy_checkpoint.pt", help="Path to save the model")
    parser.add_argument("--dataset_path", type=str, default="N/A", help="Path to load the dataset")
    parser.add_argument("--train_mode", type=str, default="single-traj", help="Options: single-traj, closest-neighbors, autoregressive, full-traj.")
    parser.add_argument("--closest_neighbors_radius", type=float, default=0.001, help="If train_mode is closest-neighbors.")
    parser.add_argument("--warm_start", type=int, default=0, help="Number of warm start epochs.")
    parser.add_argument("--train_percent", type=float, default=0.8, help="Percentage of data used for train.")
    parser.add_argument("--train_expert", action="store_true", default=False, help="Whether we're training an expert or a residual.")

    # Head architecture
    parser.add_argument("--use_new_head_arch", action="store_true", default=False, help="Whether we're using LayerNorm + SiLU + Dropout.")
    parser.add_argument("--num_head_layers", type=int, default=3, help="Number of Linear layers in the head.")
    parser.add_argument("--d_model_head", type=int, default=1024, help="Size of each Linear layer in the head.")

    # All the bounds
    parser.add_argument("--receptive_xlow", type=float, default=0.3, help="Lower bound of receptive x position.")
    parser.add_argument("--receptive_xhigh", type=float, default=0.55, help="Upper bound of receptive x position.")
    parser.add_argument("--receptive_ylow", type=float, default=-0.1, help="Lower bound of receptive y position.")
    parser.add_argument("--receptive_yhigh", type=float, default=0.5, help="Upper bound of receptive y position.")
    parser.add_argument("--insertive_xlow", type=float, default=0.3, help="Lower bound of insertive x position.")
    parser.add_argument("--insertive_xhigh", type=float, default=0.55, help="Upper bound of insertive x position.")
    parser.add_argument("--insertive_ylow", type=float, default=-0.1, help="Lower bound of insertive y position.")
    parser.add_argument("--insertive_yhigh", type=float, default=0.5, help="Upper bound of insertive y position.")
    
    args = parser.parse_args()

    # Accessing the parameters
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    D_MODEL = args.d_model
    NUM_LAYERS = args.num_layers
    DROPOUT = args.dropout

    save_path = args.save_path
    
    CONTEXT_DIM = 45 + 7
    CURRENT_DIM = 45
    LABEL_DIM = 7

    # Bounds
    RECEPTIVE_LOW = np.array([args.receptive_xlow, args.receptive_ylow])
    RECEPTIVE_HIGH = np.array([args.receptive_xhigh, args.receptive_yhigh])
    INSERTIVE_LOW = np.array([args.insertive_xlow, args.insertive_ylow])
    INSERTIVE_HIGH = np.array([args.insertive_xhigh, args.insertive_yhigh])

    if args.train_expert:
        assert args.train_mode in ["autoregressive", "full-traj"]

    if ENABLE_WANDB:
        WANDB_PROJECT = "robot-transformer-bc-deterministic-normalized-labels" if not args.train_expert else "robot-mlp-bc"
        wandb.init(project=WANDB_PROJECT, config=vars(args))
    
    DATASET_PATH = args.dataset_path

    trajs = []
    try:
        with open(DATASET_PATH, "rb") as fi:
            trajs += pickle.load(fi)
    except FileNotFoundError:
        print("Data file not found.")
        return
    print(f"Loaded dataset from {DATASET_PATH}.")

    processed_data = []
    for traj in trajs:
        if not ((traj['starting_position']['receptive_position'][:2] >= RECEPTIVE_LOW) & (traj['starting_position']['receptive_position'][:2] <= RECEPTIVE_HIGH) &
            (traj['starting_position']['insertive_position'][:2] >= INSERTIVE_LOW) & (traj['starting_position']['insertive_position'][:2] <= INSERTIVE_HIGH)).all():
            continue
        
        if traj['rewards'].ndim == 1:
            traj['rewards'] = traj['rewards'][:, None]
        
        processed_traj = {
            'context': np.concatenate([traj['obs']['policy2'], traj['actions']], axis=1),
            'current': traj['obs']['policy2'],
            'label': traj['actions_expert'] - traj['actions'],
            'choosable': traj['obs']['policy2'].shape[0] > 6,
            'obs_receptive_noise': traj['obs_receptive_noise'],
            '__log': traj,
            # 'choosable': not np.any(traj['rewards'] > 0.11),
        }
        if 'rand_noise' in traj.keys():
            traj['rand_noise'] = traj['rand_noise'].squeeze()[:processed_traj['current'].shape[0]]
            processed_traj['context'][:, CURRENT_DIM:] += traj['rand_noise']
        if args.train_expert:
            processed_traj['context'] *= 0
            processed_traj['label'] = traj['actions_expert']
        
        processed_data.append(processed_traj)
    assert processed_data[0]['context'].shape[-1] == CONTEXT_DIM
    assert processed_data[0]['current'].shape[-1] == CURRENT_DIM
    assert processed_data[0]['label'].shape[-1] == LABEL_DIM
    print(f"Kept {len(processed_data)}/{len(trajs)} ({len(processed_data)/len(trajs)}) trajectories.")

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
        'dataset_size': len(processed_data),
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

        'use_new_head_arch': args.use_new_head_arch,
        'num_head_layers': args.num_head_layers,
        'd_model_head': args.d_model_head,

        'receptive_low': RECEPTIVE_LOW,
        'receptive_high': RECEPTIVE_HIGH,
        'insertive_low': INSERTIVE_LOW,
        'insertive_high': INSERTIVE_HIGH,
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

    # Visualization
    viz_path = os.path.join(save_path, "viz")
    os.makedirs(viz_path, exist_ok=True)
    all_labels_viz = np.concatenate([traj['label'] for traj in processed_data], axis=0)
    for i in range(LABEL_DIM):
        cur_utils.save_histogram(all_labels_viz[:, i], os.path.join(viz_path, f"label_{i}.png"))
    all_receptive_locations = np.stack([traj['__log']['starting_position']['receptive_position'] for traj in processed_data], axis=0)[:, :2]
    all_insertive_locations = np.stack([traj['__log']['starting_position']['insertive_position'] for traj in processed_data], axis=0)[:, :2]
    cur_utils.save_point_distribution_image(all_receptive_locations, os.path.join(viz_path, f"loaded_receptive_locations.png"), fixed_bounds=True)
    cur_utils.save_point_distribution_image(all_insertive_locations, os.path.join(viz_path, f"loaded_insertive_locations.png"), fixed_bounds=True)

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
    model = RobotTransformerPolicy(
        CONTEXT_DIM, CURRENT_DIM, LABEL_DIM, num_layers=NUM_LAYERS, d_model=D_MODEL, dropout=DROPOUT,
        use_new_head_arch=args.use_new_head_arch,
        num_head_layers=args.num_head_layers,
        d_model_head=args.d_model_head,
    )
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
