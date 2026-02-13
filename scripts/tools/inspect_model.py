import os
import pickle
import numpy as np
import torch
import pathlib


def main():
    FILENAME = "/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/feb8/expert-ds_random5-receptive_x_geq_05-5layers_x4_relu/300-ckpt.pt"
    FILENAME = pathlib.Path(FILENAME)
    FILENAME = FILENAME.parent / "info.pkl"
    with open(FILENAME, "rb") as fi:
        info = pickle.load(fi)
    for k, v in info.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
