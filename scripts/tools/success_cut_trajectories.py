import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def main():
    FILENAME = "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/trajectories_ynnn-True-2.0-0.0-20000.pkl"
    SAVEFILE = os.path.join(os.path.dirname(FILENAME), "cut-" + os.path.basename(FILENAME))
    with open(FILENAME, "rb") as fi:
        trajs = pickle.load(fi)
    for traj in trajs:
        rewards = traj['rewards']
        if np.any(rewards > 0.11):
            cutoff = np.argmax(rewards > 0.11) + 1
        else:
            cutoff = rewards.shape[0]
        traj['rewards'] = traj['rewards'][:cutoff]
        traj['actions'] = traj['actions'][:cutoff]
        traj['actions_expert'] = traj['actions_expert'][:cutoff]
        for k in traj['obs'].keys():
            traj['obs'][k] = traj['obs'][k][:cutoff]
            traj['next_obs'][k] = traj['next_obs'][k][:cutoff]
        assert traj['dones'].sum() == 1
        traj['dones'] = traj['dones'][-cutoff:]
    with open(SAVEFILE, "wb") as fi:
        pickle.dump(trajs, fi)
    print(SAVEFILE)
    pass

if __name__ == "__main__":
    main()
