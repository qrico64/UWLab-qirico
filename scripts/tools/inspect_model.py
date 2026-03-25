import os
import pickle
import numpy as np
import torch
import pathlib


def main():
    FILENAME = "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar15/drawer_y5_id_h110/info.pkl"
    FILENAME = pathlib.Path(FILENAME)
    FILENAME = FILENAME.parent / "info.pkl"
    with open(FILENAME, "rb") as fi:
        info = pickle.load(fi)
    for k, v in info.items():
        print(f"{k}: {v}")
    with open(FILENAME.parent / "info.txt", "w") as fi:
        for k, v in info.items():
            fi.write(f"{k}: {v}\n")
    with open(FILENAME, "wb") as fi:
        pickle.dump(info, fi)

if __name__ == "__main__":
    main()
