import os
import pickle
import numpy as np
import torch
import pathlib


def main():
    FILENAME = "/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar2/experiment_s3r2_pred_ares_from_stateonly_npa_and_matchedmu_lossx1/300-ckpt.pt"
    FILENAME = pathlib.Path(FILENAME)
    FILENAME = FILENAME.parent / "info.pkl"
    with open(FILENAME, "rb") as fi:
        info = pickle.load(fi)
    for k, v in info.items():
        print(f"{k}: {v}")
    with open(FILENAME, "wb") as fi:
        pickle.dump(info, fi)

if __name__ == "__main__":
    main()
