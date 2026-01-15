# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint and record trajectories to a pkl file."""

import argparse
import sys
import pickle
import numpy as np
from tqdm import tqdm

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# NEW ARGUMENTS FOR RECORDING
parser.add_argument("--use_general_scales", action="store_true", help="Enable per-dimension noise scaling.")
parser.add_argument("--sys_noise_scale", type=float, default=0, help="Scale of system noise.")
parser.add_argument("--rand_noise_scale", type=float, default=0, help="Scale of random noise.")
parser.add_argument("--record_path", type=str, default="trajectories_ynnn.pkl", help="Path to save the recorded trajectories.")
parser.add_argument("--num_trajectories", type=int, default=10, help="Number of trajectories to record.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

def to_numpy(tensor):
    """Helper to convert torch tensors (including dicts) to numpy."""
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    return tensor.detach().cpu().numpy()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    action_shape = (env.num_envs, 7)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    
    use_general_scales = args_cli.use_general_scales
    sys_noise_scale = args_cli.sys_noise_scale
    rand_noise_scale = args_cli.rand_noise_scale
    assert sys_noise_scale >= 0 and rand_noise_scale >= 0

    if use_general_scales:
        general_noise_scales = np.array([[2.9608822, 4.3582673, 2.5497098, 8.63183, 8.950732, 2.6481836, 5.6350408]], dtype=np.float32) / 5
    else:
        general_noise_scales = np.ones((1, 7), dtype=np.float32)
    general_means = np.array([0.15601279, -0.27718163, -0.09841531, -0.5406071, -0.04196594, -0.01980124, -0.71222633], dtype=np.float32)
    savefile = args_cli.record_path
    savefile = savefile[:savefile.rfind('.')] + f"-{use_general_scales}-{sys_noise_scale}-{rand_noise_scale}-{args_cli.num_trajectories}" + savefile[savefile.rfind('.'):]
    print(f"Savefile to: {savefile}")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    # --- Recording Setup ---
    recorded_trajectories = []
    # Temp storage for currently running episodes across all envs
    # Each env index has its own dictionary of lists
    obskeys = list(obs.keys() - {'rgb'})
    current_episodes = [
        {
            "obs": {k: [] for k in obskeys},
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_obs": {k: [] for k in obskeys},
            "actions_expert": [],
            "sys_noise": np.random.normal(size=action_shape[1:], scale=sys_noise_scale),
            "starting_position": None,
        } 
        for _ in range(env.num_envs)
    ]
    total_recorded = 0

    print(f"[INFO] Recording {args_cli.num_trajectories} trajectories...")

    # Initialize Progress Bar
    print(f"Collecting data with system noise {sys_noise_scale} and random noise {rand_noise_scale}. ({use_general_scales})")
    pbar = tqdm(total=args_cli.num_trajectories, desc="Recording Trajectories", unit="traj")

    while simulation_app.is_running() and total_recorded < args_cli.num_trajectories:
        start_time = time.time()
        
        with torch.inference_mode():
            actions = policy(obs)

            actions_expert_np = to_numpy(actions)
            rand_noise = np.random.normal(size=action_shape) * rand_noise_scale
            sys_noise = np.stack([current_episodes[i]["sys_noise"] for i in range(env.num_envs)], axis=0)
            rand_noise += sys_noise
            rand_noise *= general_noise_scales
            actions += torch.tensor(rand_noise, dtype=actions.dtype, device=actions.device)
            
            # Store observations and actions (before env step)
            obs_np = to_numpy(obs)
            actions_np = to_numpy(actions)
            
            next_obs, rewards, dones, infos = env.step(actions)
            
            next_obs_np = to_numpy(next_obs)
            rewards_np = to_numpy(rewards)
            dones_np = to_numpy(dones)
            
            policy_nn.reset(dones)

            # Record transitions per environment
            for i in range(env.num_envs):
                if total_recorded >= args_cli.num_trajectories:
                    break
                
                if current_episodes[i]["starting_position"] is None:
                    obj_pos_w = env.unwrapped.scene["receptive_object"].data.root_pos_w
                    origins = env.unwrapped.scene.env_origins
                    obj_pos_in_env = obj_pos_w - origins
                    current_episodes[i]["starting_position"] = obj_pos_in_env[i].clone().detach().cpu().numpy()
                
                # Append to active buffers
                for k in obskeys:
                    current_episodes[i]["obs"][k].append(obs_np[k][i])
                    current_episodes[i]["next_obs"][k].append(next_obs_np[k][i])
                current_episodes[i]["actions"].append(actions_np[i])
                current_episodes[i]["actions_expert"].append(actions_expert_np[i])
                current_episodes[i]["rewards"].append(rewards_np[i])
                current_episodes[i]["dones"].append(dones_np[i])

                # If episode ended, finalize the trajectory
                if dones_np[i]:
                    # Convert lists to numpy arrays
                    trajectory = {
                        "obs": {k: np.array(v) for k, v in current_episodes[i]["obs"].items()},
                        "actions": np.array(current_episodes[i]["actions"]),
                        "rewards": np.array(current_episodes[i]["rewards"]),
                        "dones": np.array(current_episodes[i]["dones"]),
                        "next_obs": {k: np.array(v) for k, v in current_episodes[i]["next_obs"].items()},
                        "actions_expert": np.array(current_episodes[i]["actions_expert"]),
                        "sys_noise": np.array(current_episodes[i]["sys_noise"]),
                        "starting_position": np.array(current_episodes[i]["starting_position"]),
                    }
                    recorded_trajectories.append(trajectory)
                    pbar.update(1) # Update progress bar
                    total_recorded += 1

                    # --- Save Trajectories ---
                    if len(recorded_trajectories) % 100 == 0:
                        print(f"[INFO] Saving {len(recorded_trajectories)} trajectories to {savefile}")
                        with open(savefile, "wb") as f:
                            pickle.dump(recorded_trajectories, f)
                    
                    # Reset buffer for this env
                    current_episodes[i] = {
                        "obs": {k: [] for k in obskeys},
                        "actions": [],
                        "rewards": [],
                        "dones": [],
                        "next_obs": {k: [] for k in obskeys},
                        "actions_expert": [],
                        "sys_noise": np.random.normal(size=action_shape[1:], scale=sys_noise_scale),
                        "starting_position": None,
                    }

            obs = next_obs

        # Sleep logic for real-time
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)
    
    action_means = np.concatenate([traj['actions'] for traj in recorded_trajectories], axis=0)
    print(action_means.mean(axis=0), action_means.std(axis=0))

    # --- Save Trajectories ---
    print(f"[INFO] Saving {len(recorded_trajectories)} trajectories to {savefile}")
    with open(savefile, "wb") as f:
        pickle.dump(recorded_trajectories, f)

    pbar.close()

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
