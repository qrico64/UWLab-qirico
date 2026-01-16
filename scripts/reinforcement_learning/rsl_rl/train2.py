import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import wandb
import random


ENABLE_WANDB = False


# --- Model Definition ---

class RobotTransformerPolicy(nn.Module):
    def __init__(self, context_dim, current_dim, label_dim):
        super().__init__()
        
        # GPT-2 Small Configs
        self.n_embd = 768
        self.n_head = 12
        self.n_layer = 6
        self.dropout = 0.1
        
        self.context_proj = nn.Linear(context_dim, self.n_embd)
        self.current_proj = nn.Linear(current_dim, self.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, self.n_embd))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=self.n_head,
            dim_feedforward=4 * self.n_embd,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        
        # Deterministic Policy Head
        self.ffn = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd * 2),
            nn.GELU(),
            nn.Linear(self.n_embd * 2, self.n_embd * 2),
            nn.GELU(),
            nn.Linear(self.n_embd * 2, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, label_dim) # Direct prediction
        )

    def forward(self, context, current):
        B, T, C = context.shape
        T = T + 1
        
        # 1. Process Context and Combine with Current
        context_feat = torch.cat([self.context_proj(context), self.current_proj(current).unsqueeze(1)], dim=1)
        context_feat = context_feat + self.pos_emb[:, :T, :]
        causal = torch.triu(torch.ones(T, T, device=context.device, dtype=torch.bool), diagonal=1)
        context_out = self.transformer(context_feat, mask=causal)
        
        # 2. Extract Last Token
        context_emb = context_out[:, -1, :]
        
        # 3. Deterministic Output
        prediction = self.ffn(context_emb)
        return prediction

# --- Training and Eval Functions ---

def run_eval(model, val_data, device):
    model.eval()
    val_loss = 0
    total_steps = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for traj in val_data:
            context_seq = torch.from_numpy(traj['context']).float().to(device)
            current_vecs = torch.from_numpy(traj['current']).float().to(device)
            labels = torch.from_numpy(traj['label']).float().to(device)
            
            for t in range(1, context_seq.shape[0]):
                pred = model(context_seq[:t].unsqueeze(0), 
                             current_vecs[t].unsqueeze(0))
                
                loss = criterion(pred, labels[t].unsqueeze(0))
                val_loss += loss.item()
                total_steps += 1
                
    return val_loss / max(1, total_steps)

def train_behavior_cloning(model, train_data, val_data, epochs=100, lr=1e-4, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = 0
        total_steps = 0
        
        for traj in train_data:
            context_seq = torch.from_numpy(traj['context']).float().to(device)
            current_vecs = torch.from_numpy(traj['current']).float().to(device)
            labels = torch.from_numpy(traj['label']).float().to(device)
            
            traj_loss = 0
            for t in range(1, context_seq.shape[0]):
                pred = model(context_seq[:t].unsqueeze(0), 
                             current_vecs[t].unsqueeze(0))
                
                loss = criterion(pred, labels[t].unsqueeze(0))
                traj_loss += loss
                total_steps += 1
            
            if context_seq.shape[0] > 1:
                optimizer.zero_grad()
                # Gradient based on trajectory average
                (traj_loss / (context_seq.shape[0] - 1)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_epoch_loss += traj_loss.item()

        avg_train_loss = train_epoch_loss / max(1, total_steps)
        avg_val_loss = run_eval(model, val_data, device)
        
        print(f"Epoch {epoch+1} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
        if ENABLE_WANDB:
            wandb.log({
                "epoch": epoch + 1, 
                "train_mse": avg_train_loss, 
                "val_mse": avg_val_loss
            })

# --- Main ---

def main():
    LR = 1e-4
    EPOCHS = 100
    if ENABLE_WANDB:
        wandb.init(project="robot-transformer-bc-deterministic", config={"lr": LR, "epochs": EPOCHS})

    # Adjust dimensions here based on your data source
    CONTEXT_DIM = 45 + 7
    CURRENT_DIM = 45
    LABEL_DIM = 7
    ACTION_LOW = -0.999
    ACTION_HIGH = 0.999
    # CONTEXT_DIM = 1
    # CURRENT_DIM = 1
    # LABEL_DIM = 1

    trajs = []
    try:
        with open("trajectories_ynnn-True-0.0-0.0-10000.pkl", "rb") as fi:
            trajs += pickle.load(fi)
    except FileNotFoundError:
        print("Data file not found.")
        return
    
    
    action_means = np.concatenate([traj['actions'] for traj in trajs], axis=0)
    action_high = [8.54, 7.43, 6.33, 16.72, 30.75, 8.65, 15.46]
    action_low = [-7.84, -10.23, -7.54, -25.27, -35.54, -6.72, -16.17]
    print(f"Action high = {action_high}")
    print(f"Action low = {action_low}")
    action_high = np.array(action_high, dtype=np.float32)
    action_low = np.array(action_low, dtype=np.float32)
    assert np.all(np.quantile(action_means, 0.8, axis=0) < action_high) and np.all(np.quantile(action_means, 0.2, axis=0) > action_low)
    for traj in trajs:
        traj['actions'] = np.clip((traj['actions'] - action_low) / (action_high - action_low) * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW, ACTION_LOW, ACTION_HIGH)
        traj['actions_expert'] = np.clip((traj['actions_expert'] - action_low) / (action_high - action_low) * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW, ACTION_LOW, ACTION_HIGH)

    processed_data = []
    for traj in trajs:
        if traj['rewards'].ndim == 1:
            traj['rewards'] = traj['rewards'][:, None]
        
        processed_data.append({
            'context': np.concatenate([traj['obs']['policy2'], traj['actions']], axis=1),
            'current': traj['obs']['policy2'],
            'label': traj['actions'],
        })
        # n = traj['rewards'].shape[0]
        # arr = np.random.random(size=(n, 1))
        # processed_data.append({
        #     'context': np.random.random(size=(n, 1)),
        #     'current': np.random.random(size=(n, 1)),
        #     'label': np.random.random(size=(n, 1)),
        # })
    assert processed_data[0]['context'].shape[-1] == CONTEXT_DIM
    assert processed_data[0]['current'].shape[-1] == CURRENT_DIM
    assert processed_data[0]['label'].shape[-1] == LABEL_DIM

    random.shuffle(processed_data)
    split = int(len(processed_data) * 0.8)
    train_data = processed_data[:split]
    val_data = processed_data[split:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RobotTransformerPolicy(CONTEXT_DIM, CURRENT_DIM, LABEL_DIM)
    model.to(device)
    if ENABLE_WANDB:
        wandb.watch(model)

    try:
        train_behavior_cloning(model, train_data, val_data, epochs=EPOCHS, lr=LR, device=device)
    finally:
        if ENABLE_WANDB:
            wandb.finish()

if __name__ == '__main__':
    main()
