import os
import re

ROOT_DIR = "experiments"

# match: 1000-ckpt.pt
CKPT_PATTERN = re.compile(r"(\d+)-ckpt\.pt$")


def process_directory(dirpath):
    # --- Skip finetune directories ---
    dirname = os.path.basename(dirpath)
    if "finetune" in dirname.lower():
        print(f"Skipping (finetune dir): {dirpath}")
        return

    ckpts = []

    for fname in os.listdir(dirpath):
        match = CKPT_PATTERN.match(fname)
        if match:
            step = int(match.group(1))
            ckpts.append((step, fname))

    if len(ckpts) <= 1:
        return

    # sort by step
    ckpts.sort(key=lambda x: x[0])

    final_ckpt = ckpts[-1][1]

    for step, fname in ckpts[:-1]:
        fpath = os.path.join(dirpath, fname)
        print(f"Deleting: {fpath}")
        os.remove(fpath)


def main():
    for dirpath, _, _ in os.walk(ROOT_DIR):
        process_directory(dirpath)


if __name__ == "__main__":
    main()