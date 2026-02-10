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

def block(in_dim: int, out_dim: int, dropout: float) -> list[nn.Module]:
    return [
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
    ]

class RobotTransformerPolicy(nn.Module):
    def __init__(
            self, context_dim, current_dim, label_dim, nhead=8, num_layers=4, d_model=512, dropout=0.1,
            use_new_head_arch=False,
            num_head_layers=3,
            d_model_head=1024,
        ):
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
        if not use_new_head_arch:
            self.head = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, label_dim)
            )
        else:
            assert num_head_layers >= 3
            head_layers = block(d_model * 2, d_model_head, dropout)
            for _ in range(num_head_layers - 3):
                head_layers += block(d_model_head, d_model_head, dropout)
            head_layers += block(d_model_head, d_model, dropout)
            head_layers += [nn.Linear(d_model, label_dim)]
            self.head = nn.Sequential(*head_layers)
        print()
        print("****** Creating Transformer Policy ******")
        print("Head:")
        print(self.head)
        print("****** End Transformer Policy ******")
        print()

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
        'use_new_head_arch': False,
        'num_head_layers': 2, # intentionally wrong
        'd_model_head': 1024,
    } | save_dict
    model = RobotTransformerPolicy(
        save_dict['context_dim'],
        save_dict['current_dim'],
        save_dict['label_dim'],
        num_layers=save_dict['num_layers'],
        d_model=save_dict['d_model'],
        dropout=save_dict['dropout'],
        use_new_head_arch=save_dict['use_new_head_arch'],
        num_head_layers=save_dict['num_head_layers'],
        d_model_head=save_dict['d_model_head'],
    )
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()
    return model, save_dict

