import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, obs_dim))
        self.register_buffer("_var", torch.ones(1, obs_dim))
        self.register_buffer("_std", torch.ones(1, obs_dim))
        self.register_buffer("count", torch.tensor(0.0))

    def forward(self, x):
        return (x - self._mean) / (self._std + 0.01)


class PpoExpert(nn.Module):
    def __init__(self, our_task="peg"):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(225, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 7),
        )

        critic_in_dim = {"peg": 216, "drawer": 219, "leg": 243}[our_task]
        self.critic = nn.Sequential(
            nn.Linear(critic_in_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(64, 7))

        self.actor_obs_normalizer = EmpiricalNormalization(225)
        self.critic_obs_normalizer = EmpiricalNormalization(critic_in_dim)

    def forward(self, actor_obs):
        assert actor_obs.shape[-1] == 225, f"{actor_obs.shape} != 225"
        actor_obs = self.actor_obs_normalizer(actor_obs)
        output = self.actor(actor_obs)
        return output


def load_expert_by_path(path, device="cpu", our_task="peg"):
    ckpt = torch.load(path, map_location=device)
    model = PpoExpert(our_task=our_task).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model.eval()
    return model, ckpt

def load_expert_by_task(our_task: str, device="cpu"):
    _, _, task_expert_path = {
        "drawer": ("fbdrawerbottom", "fbdrawerbox", "expert_policies/fbdrawerbottom_state_rl_expert.pt"),
        "leg": ("fbleg", "fbtabletop", "expert_policies/fbleg_state_rl_expert.pt"),
        "peg": ("peg", "peghole", "expert_policies/peg_state_rl_expert.pt"),
    }[our_task]
    return load_expert_by_path(task_expert_path, device=device, our_task=our_task)


