# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

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
parser.add_argument("--horizon", type=int, default=60, help="Horizon, max steps, duration, whatever you call it.")
parser.add_argument("--correction_model", type=str, default="N/A", help="Residual model .pt file.")
parser.add_argument("--plot_residual", action="store_true", default=False, help="Open second screen & plot residual.")
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
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
    ManagerBasedEnv,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
import isaaclab.sim as sim_utils

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config
from train2 import RobotTransformerPolicy, load_robot_policy
import cur_utils

# PLACEHOLDER: Extension template (do not remove this comment)


def save_video(frames, path, fps=30):
    """Saves a list of frames (numpy arrays) to a video file."""
    print(f"[INFO] Saving video to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"[INFO] Video saved successfully.")

def get_positions(env: ManagerBasedEnv):
    asset = env.scene["receptive_object"]
    positions = asset.data.root_pos_w
    orientations = asset.data.root_quat_w
    asset2 = env.scene["insertive_object"]
    positions2 = asset2.data.root_pos_w
    orientations2 = asset2.data.root_quat_w
    return torch.cat([positions, orientations, positions2, orientations2], dim=-1)

def set_positions(env: ManagerBasedEnv, position: torch.Tensor, env_id: int):
    asset = env.scene["receptive_object"]
    positions = asset.data.root_pos_w
    orientations = asset.data.root_quat_w
    env_ids = torch.arange(env.num_envs, device=env.device)
    positions_orientations = torch.cat([positions, orientations], dim=-1)
    positions_orientations[env_id] = position
    asset.write_root_pose_to_sim(positions_orientations, env_ids=env_ids)

def set_positions_completely(env: ManagerBasedEnv, position: torch.Tensor, env_id: int):
    asset = env.scene["receptive_object"]
    asset.write_root_pose_to_sim(position[:7].unsqueeze(0).clone(), env_ids=torch.tensor([env_id], device=env.device))
    asset2 = env.scene["insertive_object"]
    asset2.write_root_pose_to_sim(position[7:].unsqueeze(0).clone(), env_ids=torch.tensor([env_id], device=env.device))

def render_frame(frame: np.ndarray, caption: str, display_action=None, display_action2=None):
    captions = caption.splitlines()
    IMAGE_SIZE = frame.shape[:2]
    BOUNDARY_X = 5
    BOUNDARY_Y = 5

    if display_action is not None:
        # DRAW SECOND SCREEN #
        SECOND_SCREEN_TOP_MARGIN = 40
        SECOND_SCREEN_SIZE = (IMAGE_SIZE[0] - SECOND_SCREEN_TOP_MARGIN, IMAGE_SIZE[1])
        second_screen = np.zeros((*SECOND_SCREEN_SIZE, 3), dtype=np.uint8)
        h, w = SECOND_SCREEN_SIZE
        center = (w // 2, h // 2)

        # Draw Axes (White = 255)
        cv2.line(second_screen, (0, center[1]), (w, center[1]), (255, 255, 255), 1) # X-axis
        cv2.line(second_screen, (center[0], 0), (center[0], h), (255, 255, 255), 1) # Y-axis

        # Map the -5 to +5 range to pixel coordinates
        # Scale factor: pixels per unit
        scale_x = w / (BOUNDARY_X + BOUNDARY_X)
        scale_y = h / (BOUNDARY_Y + BOUNDARY_Y)
        
        # Calculate pixel position (Note: Y is inverted in screen space)
        coord = display_action[:2]
        px = int(center[0] + coord[0] * scale_x)
        py = int(center[1] - coord[1] * scale_y)
        if abs(coord[0]) < BOUNDARY_X and abs(coord[1]) < BOUNDARY_Y:
            cv2.circle(second_screen, (px, py), 5, (0, 255, 0), -1)
        else:
            cv2.line(second_screen, center, (px, py), (0, 255, 0), 2)
        
        if display_action2 is not None:
            coord = display_action2[:2]
            px = int(center[0] + coord[0] * scale_x)
            py = int(center[1] - coord[1] * scale_y)
            if abs(coord[0]) < BOUNDARY_X and abs(coord[1]) < BOUNDARY_Y:
                cv2.circle(second_screen, (px, py), 5, (255, 0, 0), -1)
            else:
                cv2.line(second_screen, center, (px, py), (255, 0, 0), 2)
        # END DRAW SECOND SCREEN #

        top_margin = np.full((SECOND_SCREEN_TOP_MARGIN, IMAGE_SIZE[1], 3), 255, dtype=np.uint8)
        frame = np.concatenate([frame, np.concatenate([top_margin, second_screen], axis=0)], axis=1)

    cv2.putText(
        frame,
        captions[0],
        org=(5, 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(0, 0, 0),  # BGR: black
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        ' '.join(captions[1:]),
        org=(5, 28),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(0, 0, 0),  # BGR: black
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return frame

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # from source.uwlab_tasks.uwlab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCEvalCfg as ClassA
    # from uwlab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCEvalCfg as ClassB
    # a = ClassA()
    # b = ClassB()
    # print(ClassA.__module__)
    # print(ClassB.__module__)
    # import uwlab_tasks
    # import source.uwlab_tasks
    # import source.uwlab_tasks.uwlab_tasks
    # print(uwlab_tasks.__file__)
    # print(source.uwlab_tasks.__file__)
    # print(source.uwlab_tasks.uwlab_tasks.__file__)
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    print(env_cfg.to_dict()['observations'].keys())

    # set horizon
    env_cfg.episode_length_s = args_cli.horizon / 10 # don't know where the 10 came from
    print(f"Horizon: {args_cli.horizon}")

    # set camera & video
    if hasattr(args_cli, "enable_cameras") and args_cli.enable_cameras:
        IMAGE_SIZE = (400, 400)
        assert IMAGE_SIZE[0] == IMAGE_SIZE[1]
        env_cfg.scene.side_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
            update_period=0,
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.65, 0, 0.15),
                rot=(0.5, 0.5, 0.5, 0.5), # (w, x, y, z), -z direction.
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=21.9
            )
        )
        env_cfg.observations.rgb = env_cfg.observations.RGBCfg()
        env_cfg.observations.rgb.side_rgb.params['output_size'] = IMAGE_SIZE
        print(f"Video generation on at size/resolution {IMAGE_SIZE}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic
    
    RESIDUAL_S_DIM = env.observation_space['policy2'].shape[-1]
    A_DIM = env.action_space.shape[-1]
    RESIDUAL_CONTEXT_DIM = RESIDUAL_S_DIM + A_DIM
    T_DIM = args_cli.horizon
    CORRECTION_MODEL_FILE = os.path.abspath(args_cli.correction_model)
    print(f"Loading model at {CORRECTION_MODEL_FILE}")
    correction_model, correction_model_info = load_robot_policy(CORRECTION_MODEL_FILE, device=args_cli.device)
    action_multiplier = torch.tensor(correction_model_info["label_stds"], dtype=torch.float32, device=args_cli.device)
    current_means = torch.tensor(correction_model_info["current_means"], dtype=torch.float32, device=args_cli.device)
    current_stds = torch.tensor(correction_model_info["current_stds"], dtype=torch.float32, device=args_cli.device)
    context_means = torch.tensor(correction_model_info["context_means"], dtype=torch.float32, device=args_cli.device)
    context_stds = torch.tensor(correction_model_info["context_stds"], dtype=torch.float32, device=args_cli.device)
    
    assert correction_model_info['context_dim'] == RESIDUAL_S_DIM + A_DIM
    assert correction_model_info['current_dim'] == RESIDUAL_S_DIM
    assert correction_model_info['label_dim'] == A_DIM

    N_DIM = 2
    timesteps = torch.zeros(env.num_envs, N_DIM, dtype=torch.int64, device=args_cli.device)
    successes = torch.zeros(env.num_envs, N_DIM, dtype=torch.bool, device=args_cli.device)
    rec_observations = torch.zeros(env.num_envs, N_DIM, T_DIM, RESIDUAL_S_DIM, dtype=torch.float32, device=args_cli.device)
    rec_actions = torch.zeros(env.num_envs, N_DIM, T_DIM, A_DIM, dtype=torch.float32, device=args_cli.device)
    rec_rewards = torch.zeros(env.num_envs, N_DIM, T_DIM, dtype=torch.float32, device=args_cli.device)
    curstates = torch.zeros(env.num_envs, dtype=torch.int64, device=args_cli.device)
    
    obs = env.get_observations()

    if correction_model_info['obs_receptive_noise_scale'] != 0 or correction_model_info['obs_insertive_noise_scale'] != 0:
        assert 'policy_aaaaaa' in obs.keys()
    if correction_model_info['obs_insertive_noise_scale'] != 0:
        raise Exception("Right now doesn't support insertive noise!!")

    if correction_model_info['use_noise_scales']:
        general_noise_scales = torch.tensor([[2.9608822, 4.3582673, 2.5497098, 8.63183, 8.950732, 2.6481836, 5.6350408]], dtype=torch.float32, device=args_cli.device) / 5
    else:
        general_noise_scales = torch.ones(1, 7, dtype=torch.float32, device=args_cli.device)
    sys_noises = torch.randn(env.num_envs, A_DIM, device=args_cli.device) * correction_model_info['sys_noise_scale'] * general_noise_scales
    print(f"Using systematic noise of {correction_model_info['sys_noise_scale']}")
    obs_receptive_noise = torch.cat([torch.randn(env.num_envs, 2, device=args_cli.device) * correction_model_info['obs_receptive_noise_scale'], torch.zeros(env.num_envs, 4, device=args_cli.device)], dim=-1)
    print(f"Using obs_receptive_noise of {correction_model_info['obs_receptive_noise_scale']}")
    obs_insertive_noise = torch.cat([torch.randn(env.num_envs, 2, device=args_cli.device) * correction_model_info['obs_insertive_noise_scale'], torch.zeros(env.num_envs, 4, device=args_cli.device)], dim=-1)
    print(f"Using obs_insertive_noise of {correction_model_info['obs_insertive_noise_scale']}")

    if args_cli.enable_cameras:
        PLOT_RESIDUAL = args_cli.plot_residual
        rec_video = np.zeros((env.num_envs, 2, T_DIM, IMAGE_SIZE[0], IMAGE_SIZE[1] * 2 if PLOT_RESIDUAL else IMAGE_SIZE[1], 3), dtype=np.uint8)
        VIDEO_PATH = "./viz/test/video.mp4"
        videopath_generator = lambda x, y: VIDEO_PATH[:VIDEO_PATH.rfind('.')] + f"_{x}_{y}" + VIDEO_PATH[VIDEO_PATH.rfind('.'):]
        NUM_VIDEOS = 6
        VIDEO_FPS = 5
        count_success_first_try_video = 0
        count_success_second_try_video = 0
        count_failed_video = 0
    
    starting_positions = get_positions(env.env.env)

    # reset environment

    global_timestep = 0
    count_completed = 0
    count_success_first_try = 0
    count_success_second_try = 0
    while count_completed < 1000:
        global_timestep += 1
        with torch.inference_mode():
            expert_actions = policy(obs)

            obs_tweaked = obs.clone()
            receptive_noise = obs_receptive_noise
            insertive_noise = obs_insertive_noise
            receptive_state = obs_tweaked['policy_aaaaaa']['receptive_asset_pose'].reshape(env.num_envs, 5, 6) + receptive_noise.unsqueeze(1)
            insertive_state = obs_tweaked['policy_aaaaaa']['insertive_asset_pose'].reshape(env.num_envs, 5, 6) + insertive_noise.unsqueeze(1)
            obs_tweaked['policy'][:, :30] = cur_utils.predict_relative_pose(insertive_state.reshape(-1, 6), receptive_state.reshape(-1, 6)).reshape(env.num_envs, 30)
            obs_tweaked['policy'][:, -30:] = receptive_state.reshape(env.num_envs, 30)

            base_actions_raw = policy(obs_tweaked)

            base_actions_raw += sys_noises
            base_actions = base_actions_raw.clone()

            need_residuals = curstates > 0
            need_residuals_count = need_residuals.sum()
            if need_residuals_count > 0:
                contexts = torch.cat([rec_observations[need_residuals, 0, :, :], rec_actions[need_residuals, 0, :, :]], dim=2)
                currents = obs['policy2'][need_residuals].clone()
                padding_mask = torch.arange(T_DIM, device=args_cli.device).repeat(need_residuals_count, 1) >= timesteps[need_residuals, 0].unsqueeze(1)
                contexts = (contexts - context_means) / context_stds
                currents = (currents - current_means) / current_stds
                residual_actions = correction_model(contexts, currents, padding_mask)
                base_actions[need_residuals, :] += residual_actions * action_multiplier
            
            # step
            next_obs, reward, dones, info = env.step(base_actions)

            # handle non-dones
            indices = torch.arange(env.num_envs, device=args_cli.device)
            cur_timesteps = timesteps[indices, curstates]
            rec_observations[indices, curstates, cur_timesteps, :] = obs['policy2']
            rec_actions[indices, curstates, cur_timesteps, :] = base_actions
            rec_rewards[indices, curstates, cur_timesteps] = reward
            successes[indices, curstates] |= reward > 0.11

            if args_cli.enable_cameras:
                frames = obs['rgb'].cpu().detach().numpy().transpose(0, 2, 3, 1)
                if frames.max() <= 1.0 + 1e-4:
                    frames = (frames * 255).astype(np.uint8)
                
                for i in range(env.num_envs):
                    assert frames[i].shape == (*IMAGE_SIZE, 3)
                    display_action = expert_actions[i] - base_actions_raw[i] if PLOT_RESIDUAL else None
                    display_action2 = base_actions[i] - base_actions_raw[i] if PLOT_RESIDUAL else None
                    caption = f"t={timesteps[i].tolist()} r={reward[i]:.5f} done={dones[i]}"
                    if PLOT_RESIDUAL:
                        caption += f" residual-action={display_action}"
                    caption += f"\nnoise={obs_receptive_noise[i]}"
                    final_screen = render_frame(frames[i], caption, display_action=display_action, display_action2=display_action2)
                    rec_video[i, curstates[i], timesteps[i, curstates[i]]] = final_screen
            
            timesteps[indices, curstates] += 1
            obs = next_obs

            # handle dones
            policy_nn.reset(dones)
            if dones.sum() > 0:
                for i in range(env.num_envs):
                    if not dones[i]:
                        continue
                    if curstates[i] > 0 or successes[i, curstates[i]]:
                        count_completed += 1
                        count_success_first_try += curstates[i] == 0 and successes[i, curstates[i]]
                        count_success_second_try += curstates[i] == 1 and successes[i, curstates[i]]
                        # print(f"Environment {i} has finished with {['fail', 'success'][successes[i]]} at stage {curstates[i]}")

                        if args_cli.enable_cameras:
                            if curstates[i] == 0 and successes[i, curstates[i]]:
                                # first try success
                                if count_success_first_try_video < NUM_VIDEOS:
                                    videopath = videopath_generator(0, count_success_first_try_video)
                                    frames = rec_video[i, curstates[i], :timesteps[i, curstates[i]]]
                                    save_video(frames, videopath, fps=VIDEO_FPS)
                                    count_success_first_try_video += 1
                            elif curstates[i] == 1 and successes[i, curstates[i]]:
                                # second try success
                                if count_success_second_try_video < NUM_VIDEOS:
                                    videopath = videopath_generator(1, count_success_second_try_video)
                                    frames = np.concatenate([rec_video[i, 0, :timesteps[i, 0]], rec_video[i, 1, :timesteps[i, 1]]], axis=0)
                                    save_video(frames, videopath, fps=VIDEO_FPS)
                                    count_success_second_try_video += 1
                            else:
                                # failed
                                if count_failed_video < NUM_VIDEOS:
                                    videopath = videopath_generator(-1, count_failed_video)
                                    frames = np.concatenate([rec_video[i, 0, :timesteps[i, 0]], rec_video[i, 1, :timesteps[i, 1]]], axis=0)
                                    save_video(frames, videopath, fps=VIDEO_FPS)
                                    count_failed_video += 1

                        if count_completed % 20 == 0:
                            print(f"First try success rate: {count_success_first_try / count_completed}; Second try success rate: {count_success_second_try / (count_completed - count_success_first_try)}")
                            print(f"{count_completed} {count_success_first_try} {count_success_second_try}")
                            print()

                        rec_observations[i] *= 0
                        rec_actions[i] *= 0
                        rec_rewards[i] *= 0
                        timesteps[i] *= 0
                        successes[i] = False
                        curstates[i] *= 0
                        sys_noises[i] = torch.randn(A_DIM, device=args_cli.device) * correction_model_info['sys_noise_scale'] * general_noise_scales
                        obs_receptive_noise[i] = torch.cat([torch.randn(2, device=args_cli.device) * correction_model_info['obs_receptive_noise_scale'], torch.zeros(4, device=args_cli.device)], dim=-1)
                        obs_insertive_noise[i] = torch.cat([torch.randn(2, device=args_cli.device) * correction_model_info['obs_insertive_noise_scale'], torch.zeros(4, device=args_cli.device)], dim=-1)
                        starting_positions[i] = get_positions(env.env.env)[i]
                    else:
                        curstates[i] += 1
                        set_positions_completely(env.env.env, starting_positions[i], i)
        
        if args_cli.enable_cameras and count_success_first_try_video >= NUM_VIDEOS and count_success_second_try_video >= NUM_VIDEOS and count_failed_video >= NUM_VIDEOS:
            break
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
