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
import expert_utils


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
        elif train_mode == "perfect-coverage":
            self.choosable_trajs = [traj for traj in data if traj.get('choosable', False)]
        elif train_mode == "expert":
            self.choosable_trajs = []
            self.context_dim = data[0]['context'].shape[-1]
            self.current_dim = data[0]['current'].shape[-1]
            self.action_dim = data[0]['expert_actions'].shape[-1]
            self.choosable_currents = np.concatenate([traj['current'] for traj in data if traj.get('choosable', False)], axis=0)
            self.corresponding_expert_actions = np.concatenate([traj['expert_actions'] for traj in data if traj.get('choosable', False)], axis=0)
        else:
            raise NotImplementedError(train_mode)

    def __len__(self):
        if self.train_mode == "expert":
            return self.choosable_currents.shape[0]
        else:
            return len(self.choosable_trajs)

    def __getitem__(self, idx):
        if self.train_mode == "expert":
            fake_context = torch.zeros(1, self.context_dim, dtype=torch.float32)
            current = torch.tensor(self.choosable_currents[idx], dtype=torch.float32)
            fake_base_action = torch.zeros(self.action_dim, dtype=torch.float32)
            expert_action = torch.tensor(self.corresponding_expert_actions[idx], dtype=torch.float32)
            return fake_context, current, fake_base_action, expert_action

        # Get the context and label from a "choosable" trajectory
        traj = self.choosable_trajs[idx]

        sys_noise = torch.tensor(traj['sys_noise'], dtype=torch.float32)
        obs_noise = torch.tensor(traj['obs_receptive_noise'], dtype=torch.float32)
        _ref_traj = None

        if self.train_mode == "closest-neighbors":
            context = torch.tensor(traj['context'], dtype=torch.float32)
            second_traj = self.data[np.random.choice(self.valid_seconds[idx])]
            st = random.randint(0, second_traj['current'].shape[0] - 1)
            current = torch.tensor(second_traj['current'][st], dtype=torch.float32)
            base_action = torch.tensor(second_traj['base_actions'][st], dtype=torch.float32)
            expert_action = torch.tensor(second_traj['expert_actions'][st], dtype=torch.float32)
        elif self.train_mode == "single-traj":
            T = traj['context'].shape[0]
            assert T > 6, f"{T}"
            zt = random.randint(6, T - 1)
            st = random.randint(zt, T - 1)
            context = torch.tensor(traj['context'][:zt], dtype=torch.float32)
            current = torch.tensor(traj['current'][st], dtype=torch.float32)
            base_action = torch.tensor(traj['base_actions'][st], dtype=torch.float32)
            expert_action = torch.tensor(traj['expert_actions'][st], dtype=torch.float32)
        elif self.train_mode == "autoregressive":
            T = traj['context'].shape[0]
            assert T > 6, f"{T}"
            zt = random.randint(1, T - 1)
            context = torch.tensor(traj['context'][:zt], dtype=torch.float32)
            current = torch.tensor(traj['current'][zt], dtype=torch.float32)
            base_action = torch.tensor(traj['base_actions'][zt], dtype=torch.float32)
            expert_action = torch.tensor(traj['expert_actions'][zt], dtype=torch.float32)
        elif self.train_mode == "full-traj":
            T = traj['current'].shape[0]
            assert T > 6, f"{T}"
            zt = random.randint(0, T - 1)
            context = torch.tensor(traj['context'], dtype=torch.float32)
            current = torch.tensor(traj['current'][zt], dtype=torch.float32)
            base_action = torch.tensor(traj['base_actions'][zt], dtype=torch.float32)
            expert_action = torch.tensor(traj['expert_actions'][zt], dtype=torch.float32)
        elif self.train_mode == "perfect-coverage":
            context = torch.tensor(traj['context'], dtype=torch.float32)
            si = random.randint(0, len(self.choosable_trajs) - 1)
            second_traj = self.choosable_trajs[si]
            st = random.randint(0, second_traj['current'].shape[0] - 1)
            current = torch.tensor(second_traj['current'][st], dtype=torch.float32)
            base_action = torch.tensor(second_traj['base_actions'][st], dtype=torch.float32)
            expert_action = torch.tensor(second_traj['expert_actions'][st], dtype=torch.float32)
            _ref_traj = (
                second_traj['__log']['obs']['policy'][st],
                second_traj['__log']['obs']['policy_aaaaaa']['receptive_asset_pose'][st],
                second_traj['__log']['obs']['policy_aaaaaa']['insertive_asset_pose'][st],
            )
        else:
            raise NotImplementedError(self.train_mode)
        
        data_source = traj['data_source']
        
        return context, current, base_action, expert_action, data_source, sys_noise, obs_noise, _ref_traj

def collate_fn(batch):
    """
    Custom collator to pad trajectories of different lengths.
    """
    contexts, currents, base_actions, expert_actions, data_sources, sys_noises, obs_noises, _ref_trajs = zip(*batch)
    
    # Pad sequences to the max length in this specific batch
    # padded_contexts shape: (Batch, Max_T, Context_Dim)
    padded_contexts = torch.nn.utils.rnn.pad_sequence(contexts, batch_first=True)
    
    # Create a mask: True for padded positions, False for real data
    # This is for PyTorch's src_key_padding_mask
    padding_mask = torch.zeros(padded_contexts.shape[0], padded_contexts.shape[1], dtype=torch.bool)
    for i, ctx in enumerate(contexts):
        padding_mask[i, len(ctx):] = True
        
    currents = torch.stack(currents)
    base_actions = torch.stack(base_actions)
    expert_actions = torch.stack(expert_actions)
    sys_noises = torch.stack(sys_noises)
    obs_noises = torch.stack(obs_noises)
    
    return padded_contexts, currents, base_actions, expert_actions, padding_mask, data_sources, sys_noises, obs_noises, _ref_trajs

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
        ref_label_means = None,
        ref_label_stds = None,
        ref_current_means = None,
        ref_current_stds = None,
    ):
    unique_data_sources_train = {}
    for traj in train_data:
        unique_data_sources_train[traj['data_source']] = unique_data_sources_train.get(traj['data_source'], 0) + 1
    unique_data_sources_val = {}
    for traj in val_data:
        unique_data_sources_val[traj['data_source']] = unique_data_sources_val.get(traj['data_source'], 0) + 1

    expert_model = expert_utils.load_peg_expert("peg_state_rl_expert.pt", device='cuda')[0]
    assert np.allclose(
        train_data[0]['expert_actions'], 
        (expert_model(torch.tensor(train_data[0]['__log']['obs']['policy'], device=device)).cpu().detach().numpy() - ref_label_means) / ref_label_stds,
        atol=1e-5,
    )
    assert np.allclose(
        train_data[0]['current'][0][:6],
        (train_data[0]['__log']['obs']['policy'][0][:6] - ref_current_means[:6]) / ref_current_stds[:6],
        rtol=1e-4,
    )
    assert np.allclose(
        train_data[0]['current'][0][39:45],
        (train_data[0]['__log']['obs']['policy'][0][195:201] - ref_current_means[39:45]) / ref_current_stds[39:45],
        rtol=1e-4,
    )

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

    assert epochs % 5 == 0, f"epochs={epochs} must be divisible by 5"
    SAVE_INTERVAL = epochs // 5

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
        total_info = {}
        
        for context, current, base_actions, expert_actions, padding_mask, data_sources, sys_noises, obs_noises, _ref_trajs in pbar:
            context, current, base_actions, expert_actions = context.to(device), current.to(device), base_actions.to(device), expert_actions.to(device)
            padding_mask = padding_mask.to(device)
            sys_noises = sys_noises.to(device)
            obs_noises = obs_noises.to(device)

            if train_mode == "perfect-coverage":
                assert _ref_trajs[0] is not None
                _ref_label_means = torch.tensor(ref_label_means, dtype=torch.float32, device=device)
                _ref_label_stds = torch.tensor(ref_label_stds, dtype=torch.float32, device=device)
                _ref_obss = torch.stack([torch.tensor(rt[0], dtype=torch.float32, device=device) for rt in _ref_trajs])
                _ref_recposes = torch.stack([torch.tensor(rt[1], dtype=torch.float32, device=device) for rt in _ref_trajs])
                _ref_insposes = torch.stack([torch.tensor(rt[2], dtype=torch.float32, device=device) for rt in _ref_trajs])
                _ref_noised_obss = cur_utils.apply_obs_noise2(_ref_obss, _ref_recposes, _ref_insposes, obs_noises.cpu().numpy())
                _ref_new_actions = (expert_model(_ref_noised_obss) - _ref_label_means) / _ref_label_stds
                base_actions = _ref_new_actions.to(device)

            optimizer.zero_grad()
            loss, info = model.loss(context, current, base_actions, expert_actions, padding_mask=padding_mask)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            for data_source in unique_data_sources_train.keys():
                correspondent = np.array([ds == data_source for ds in data_sources])
                if correspondent.sum() <= 0:
                    continue
                for k, v in info.items():
                    if isinstance(v, np.ndarray):
                        total_info[f"{data_source}/{k}"] = total_info.get(f"{data_source}/{k}", 0) + v[correspondent].sum()
                    else:
                        total_info[f"{data_source}/{k}"] = total_info.get(f"{data_source}/{k}", 0) + v * correspondent.sum()

        # Validation phase
        model.eval()
        val_loss = 0
        total_vinfo = {}
        with torch.no_grad():
            for context, current, base_actions, expert_actions, padding_mask, data_sources, sys_noises, obs_noises, _ref_trajs in val_loader:
                context, current, base_actions, expert_actions = context.to(device), current.to(device), base_actions.to(device), expert_actions.to(device)
                padding_mask = padding_mask.to(device)
                sys_noises = sys_noises.to(device)
                obs_noises = obs_noises.to(device)

                if train_mode == "perfect-coverage":
                    assert _ref_trajs[0] is not None
                    _ref_label_means = torch.tensor(ref_label_means, dtype=torch.float32, device=device)
                    _ref_label_stds = torch.tensor(ref_label_stds, dtype=torch.float32, device=device)
                    _ref_obss = torch.stack([torch.tensor(rt[0], dtype=torch.float32, device=device) for rt in _ref_trajs])
                    _ref_recposes = torch.stack([torch.tensor(rt[1], dtype=torch.float32, device=device) for rt in _ref_trajs])
                    _ref_insposes = torch.stack([torch.tensor(rt[2], dtype=torch.float32, device=device) for rt in _ref_trajs])
                    _ref_noised_obss = cur_utils.apply_obs_noise2(_ref_obss, _ref_recposes, _ref_insposes, obs_noises.cpu().numpy())
                    _ref_new_actions = (expert_model(_ref_noised_obss) - _ref_label_means) / _ref_label_stds
                    base_actions = _ref_new_actions.to(device)

                vloss, vinfo = model.loss(context, current, base_actions, expert_actions, padding_mask=padding_mask)
                val_loss += vloss.item()

                for data_source in unique_data_sources_val.keys():
                    correspondent = np.array([ds == data_source for ds in data_sources])
                    if correspondent.sum() <= 0:
                        continue
                    for k, v in vinfo.items():
                        if isinstance(v, np.ndarray):
                            total_vinfo[f"{data_source}/{k}"] = total_vinfo.get(f"{data_source}/{k}", 0) + v[correspondent].sum()
                        else:
                            total_vinfo[f"{data_source}/{k}"] = total_vinfo.get(f"{data_source}/{k}", 0) + v * correspondent.sum()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Summary - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if ENABLE_WANDB:
            wandblog = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr'],
            }
            for k in total_info.keys():
                data_source, metric_name = k[:k.find('/')], k[k.find('/')+1:]
                wandblog[f"train_{data_source}/{metric_name}"] = total_info[k] / unique_data_sources_train[data_source]
            for k in total_vinfo.keys():
                data_source, metric_name = k[:k.find('/')], k[k.find('/')+1:]
                wandblog[f"val_{data_source}/{metric_name}"] = total_vinfo[k] / unique_data_sources_val[data_source]
            wandb.log(wandblog)
        
        if (epoch + 1) % SAVE_INTERVAL == 0 and save_path is not None:
            csp = os.path.join(save_path, f"{epoch}-ckpt.pt")
            torch.save(model.state_dict(), csp)
            print(f"Model at epoch {epoch} saved to {csp}")
            fixed_epochs.append(epoch)
        
        if epoch > SAVE_INTERVAL - 10 and avg_val_loss < best_loss and save_path is not None and epoch not in fixed_epochs:
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
    parser.add_argument("--dataset_path", type=str, nargs='+', default=["N/A"], help="Path(s) to load the dataset (multiple paths will be combined)")
    parser.add_argument("--train_mode", type=str, default="single-traj", help="Options: single-traj, closest-neighbors, autoregressive, full-traj.")
    parser.add_argument("--closest_neighbors_radius", type=float, default=0.001, help="If train_mode is closest-neighbors.")
    parser.add_argument("--warm_start", type=int, default=0, help="Number of warm start epochs.")
    parser.add_argument("--train_percent", type=float, default=0.8, help="Percentage of data used for train.")
    parser.add_argument("--infer_mode", type=str, default="residual", help="Options: residual, expert, res_scale_shift.")
    parser.add_argument("--state_type", type=str, default="standard", help="Options: standard, noprevaction, eeposition, perfectmu, state_baseaction, baseaction_only.")

    # Mu stuff
    parser.add_argument("--mu_head_arch", type=str, default="none", help="Options: none, identity, linear, 2layer.")
    parser.add_argument("--mu_size", type=int, default=512, help="Dimension of mu.")
    parser.add_argument("--mu_kl_factor", type=float, default=0.0, help="KL factor for mu.")

    # Curr KL stuff
    parser.add_argument("--current_norm", action="store_true", help="Whether to apply layer normalization to current embeddings.")
    parser.add_argument("--current_head_arch", type=str, default="none", help="Options: none, linear.")
    parser.add_argument("--current_emb_size", type=int, default=512, help="Dimension of current.")
    parser.add_argument("--current_kl_factor", type=float, default=0.0, help="KL factor for current.")

    # Combined head stuff
    parser.add_argument("--combined_head_arch", type=str, default="none", help="Options: none, linear, 2layer.")
    parser.add_argument("--combined_emb_size", type=int, default=512, help="Dimension of combined.")
    parser.add_argument("--combined_kl_factor", type=float, default=0.0, help="KL factor for combined.")

    # Head architecture
    parser.add_argument("--head_arch_version", type=str, default="ancient", help="Options: ancient, blocked, mlpblock_v1.")
    parser.add_argument("--num_head_layers", type=int, default=3, help="Number of Linear layers in the head.")
    parser.add_argument("--d_model_head", type=int, default=1024, help="Size of each Linear layer in the head.")
    parser.add_argument("--dropout_head", type=float, default=0.0, help="Dropout rate for head layers.")

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

    TRAIN_EXPERT = args.infer_mode in ["expert", "expert_new"]
    if TRAIN_EXPERT:
        assert args.infer_mode == "expert_new", "expert is deprecated."
        assert args.train_mode in ["expert"]

    if ENABLE_WANDB:
        WANDB_PROJECT = "robot-transformer-bc-deterministic-normalized-labels" if not TRAIN_EXPERT else "robot-mlp-bc"
        WANDB_NAME = os.path.basename(save_path)
        wandb.init(project=WANDB_PROJECT, config=vars(args), name=WANDB_NAME)
    
    DATASET_PATHS = args.dataset_path

    DATASET_NAMES = {
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb26/fourthtry_receptive_0_sys3_rand2_recxgeq05/job-True-3.0-2.0-100000-60--0.0-0.0/cut-trajectories.pkl": "sysnoise_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb17/fourthtry_receptive_0.01_with_randnoise_2.0_recxgeq05/job-True-0.0-2.0-100000-60--0.01-0.0/cut-trajectories.pkl": "obsnoise_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb19/fourthtry_receptive_0006_sys4_rand2_recxgeq05/job-True-4.0-2.0-100000-60--0.006-0.0/cut-trajectories.pkl": "obs0006_sys4_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar5/obs001r2_dataset_recxgeq05/job-True-0.0-2.0-100000-60--0.01-0.0/cut-trajectories.pkl": "obsnoise_ds_new",
    }

    datasets = []
    total_trajs = 0
    for DATASET_PATH in DATASET_PATHS:
        assert DATASET_PATH in DATASET_NAMES
        try:
            with open(DATASET_PATH, "rb") as fi:
                loaded = pickle.load(fi)
                datasets.append((loaded, DATASET_PATH))
                total_trajs += len(loaded)
                print(f"Loaded {len(loaded)} trajectories from {DATASET_PATH}.")
        except FileNotFoundError:
            print(f"Data file not found: {DATASET_PATH}")
            return

    processed_data = []
    for dataset in datasets:
        trajs, data_source = dataset
        dataset_name = DATASET_NAMES.get(data_source, "unknown_ds")
        for traj in trajs:
            if not ((traj['starting_position']['receptive_position'][:2] >= RECEPTIVE_LOW) & (traj['starting_position']['receptive_position'][:2] <= RECEPTIVE_HIGH) &
                (traj['starting_position']['insertive_position'][:2] >= INSERTIVE_LOW) & (traj['starting_position']['insertive_position'][:2] <= INSERTIVE_HIGH)).all():
                continue
            
            if traj['rewards'].ndim == 1:
                traj['rewards'] = traj['rewards'][:, None]
            
            processed_traj = {
                'context': np.concatenate([traj['obs']['policy2'], traj['actions']], axis=1),
                'current': np.concatenate([traj['obs']['policy2'], traj['actions']], axis=1),
                'base_actions': traj['actions'],
                'expert_actions': traj['actions_expert'],
                'choosable': traj['obs']['policy2'].shape[0] > 6,
                'obs_receptive_noise': traj['obs_receptive_noise'],
                'sys_noise': traj['sys_noise'],
                'data_source': dataset_name,
                '__log': traj,
                # 'choosable': not np.any(traj['rewards'] > 0.11),
            }
            if 'rand_noise' in traj.keys():
                traj['rand_noise'] = traj['rand_noise'].squeeze()[:processed_traj['current'].shape[0]]
                processed_traj['context'][:, -LABEL_DIM:] += traj['rand_noise']
            
            processed_data.append(processed_traj)
    assert processed_data[0]['context'].shape[-1] == CONTEXT_DIM
    print(f"Kept {len(processed_data)}/{total_trajs} ({len(processed_data)/total_trajs}) trajectories.")

    # Current normalization
    all_currents = np.concatenate([traj['current'] for traj in processed_data], axis=0)
    current_means = all_currents.mean(axis=0)
    current_stds = all_currents.std(axis=0)
    all_contexts = np.concatenate([traj['context'] for traj in processed_data], axis=0)
    context_means = all_contexts.mean(axis=0)
    context_stds = all_contexts.std(axis=0) + 1e-9
    if TRAIN_EXPERT:
        all_labels = np.concatenate([traj['expert_actions'] for traj in processed_data], axis=0)
    elif args.infer_mode == "res_scale_shift":
        all_labels = np.concatenate([traj['expert_actions'] for traj in processed_data] + [traj['base_actions'] for traj in processed_data], axis=0)
    elif args.infer_mode == "residual":
        all_labels = np.concatenate([traj['expert_actions'] - traj['base_actions'] for traj in processed_data], axis=0)
    label_means = all_labels.mean(axis=0)
    label_stds = all_labels.std(axis=0)
    for traj in processed_data:
        traj['current'] = (traj['current'] - current_means) / current_stds
        traj['context'] = (traj['context'] - context_means) / context_stds
        if args.infer_mode == "res_scale_shift":
            traj['base_actions'] = (traj['base_actions'] - label_means) / label_stds
        else:
            traj['base_actions'] = traj['base_actions'] / label_stds
        traj['expert_actions'] = (traj['expert_actions'] - label_means) / label_stds

    save_dict = {
        'dataset_origin': [os.path.abspath(p) for p in DATASET_PATHS],
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
        'train_expert': TRAIN_EXPERT,
        'infer_mode': args.infer_mode,
        'mu_head_arch': args.mu_head_arch,
        'mu_size': args.mu_size,
        'mu_kl_factor': args.mu_kl_factor,

        'head_arch_version': args.head_arch_version,
        'num_head_layers': args.num_head_layers,
        'd_model_head': args.d_model_head,
        'dropout_head': args.dropout_head,

        'receptive_low': RECEPTIVE_LOW,
        'receptive_high': RECEPTIVE_HIGH,
        'insertive_low': INSERTIVE_LOW,
        'insertive_high': INSERTIVE_HIGH,

        'current_norm': args.current_norm,
        'current_head_arch': args.current_head_arch,
        'current_emb_size': args.current_emb_size,
        'current_kl_factor': args.current_kl_factor,

        'combined_head_arch': args.combined_head_arch,
        'combined_emb_size': args.combined_emb_size,
        'combined_kl_factor': args.combined_kl_factor,

        'state_type': args.state_type,
    }
    # Use the first dataset path for info.pkl lookup (noise scale metadata)
    first_path = DATASET_PATHS[0]
    with open(os.path.join(os.path.dirname(first_path), "info.pkl"), "rb") as fi:
        load_dict = pickle.load(fi)
    save_dict |= {
        'use_noise_scales': load_dict['use_general_scales'],
        'sys_noise_scale': load_dict['sys_noise_scale'],
        'rand_noise_scale': load_dict['rand_noise_scale'],
        'obs_insertive_noise_scale': load_dict['obs_insertive_noise_scale'],
        'obs_receptive_noise_scale': load_dict['obs_receptive_noise_scale'],
    }
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "info.pkl"), "wb") as fi:
        pickle.dump(save_dict, fi)

    # Visualization
    viz_path = os.path.join(save_path, "viz")
    os.makedirs(viz_path, exist_ok=True)
    all_base_actions_viz = np.concatenate([traj['base_actions'] for traj in processed_data], axis=0)
    for i in range(LABEL_DIM):
        cur_utils.save_histogram(all_base_actions_viz[:, i], os.path.join(viz_path, f"base_action_{i}.png"))
    all_expert_actions_viz = np.concatenate([traj['expert_actions'] for traj in processed_data], axis=0)
    for i in range(LABEL_DIM):
        cur_utils.save_histogram(all_expert_actions_viz[:, i], os.path.join(viz_path, f"expert_action_{i}.png"))
    all_residual_actions_viz = np.concatenate([traj['expert_actions'] - traj['base_actions'] for traj in processed_data], axis=0)
    for i in range(LABEL_DIM):
        cur_utils.save_histogram(all_residual_actions_viz[:, i], os.path.join(viz_path, f"residual_action_{i}.png"))
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
        head_arch_version=args.head_arch_version,
        num_head_layers=args.num_head_layers,
        d_model_head=args.d_model_head,
        dropout_head=args.dropout_head,
        infer_mode=args.infer_mode,
        mu_head_arch=args.mu_head_arch,
        mu_size=args.mu_size,
        mu_kl_factor=args.mu_kl_factor,
        current_norm=args.current_norm,
        current_head_arch=args.current_head_arch,
        current_emb_size=args.current_emb_size,
        current_kl_factor=args.current_kl_factor,
        combined_head_arch=args.combined_head_arch,
        combined_emb_size=args.combined_emb_size,
        combined_kl_factor=args.combined_kl_factor,
        state_type=args.state_type,
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
            ref_label_means=label_means,
            ref_label_stds=label_stds,
            ref_current_means=current_means,
            ref_current_stds=current_stds,
        )
    finally:
        if ENABLE_WANDB:
            wandb.finish()

if __name__ == '__main__':
    main()
