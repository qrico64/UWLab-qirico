import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def main():
    FILENAME = "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar15/leg_y5_id_h160/trajectories.pkl"
    SAVEFILE = os.path.join(os.path.dirname(FILENAME), "cut-" + os.path.basename(FILENAME))
    with open(FILENAME, "rb") as fi:
        trajs = pickle.load(fi)
    with open(os.path.join(os.path.dirname(FILENAME), "info.pkl"), "rb") as fi:
        info = pickle.load(fi)
    result_trajs = []
    success_count = 0
    elimination_count = 0
    horizon = max([traj['actions'].shape[0] for traj in trajs])
    for traj in trajs:
        max_action_magnitude = np.linalg.norm(traj['actions'], axis=-1).max()
        if max_action_magnitude > 100 or traj['actions'].shape[0] < horizon:
            elimination_count += 1
            continue
        
        rewards = traj['rewards']
        if np.any(rewards > 0.09):
            success_count += 1
            cutoff = np.argmax(rewards > 0.09) + 1
        else:
            cutoff = rewards.shape[0]
        traj['rewards'] = traj['rewards'][:cutoff]
        traj['actions'] = traj['actions'][:cutoff]
        traj['actions_expert'] = traj['actions_expert'][:cutoff]
        for k in traj['obs'].keys() - {'policy_aaaaaa'}:
            traj['obs'][k] = traj['obs'][k][:cutoff]
            traj['next_obs'][k] = traj['next_obs'][k][:cutoff]
        if 'policy_aaaaaa' in traj['obs'].keys():
            for k in traj['obs']['policy_aaaaaa'].keys():
                traj['obs']['policy_aaaaaa'][k] = traj['obs']['policy_aaaaaa'][k][:cutoff]
                traj['next_obs']['policy_aaaaaa'][k] = traj['next_obs']['policy_aaaaaa'][k][:cutoff]
        assert traj['dones'].sum() == 1
        traj['dones'] = traj['dones'][-cutoff:]
        if 'rand_noise' in traj:
            traj['rand_noise'] = traj['rand_noise'][:cutoff]
        result_trajs.append(traj)
    
    print(f"Elimination rate: {(len(trajs) - len(result_trajs)) / len(trajs)}")
    print(f"Success rate: {success_count / len(result_trajs)}")

    info['after_cut_success_rate'] = success_count / len(result_trajs)
    info['cut_elimination_rate'] = (len(trajs) - len(result_trajs)) / len(trajs)
    with open(os.path.join(os.path.dirname(SAVEFILE), "info.pkl"), "wb") as fi:
        pickle.dump(info, fi)
    with open(os.path.join(os.path.dirname(SAVEFILE), "info.txt"), "w") as fi:
        for k, v in info.items():
            fi.write(f"{k}: {v}\n")
    with open(SAVEFILE, "wb") as fi:
        pickle.dump(result_trajs, fi)
    print(SAVEFILE)
    pass

if __name__ == "__main__":
    main()
