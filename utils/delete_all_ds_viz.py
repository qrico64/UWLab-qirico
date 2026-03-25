import os

# --- CONFIG ---
ROOT_DIR = "experiments"  # change or pass via CLI if needed
DRY_RUN = False  # set to False to actually delete

# Target filenames
TARGET_PREFIXES = [
    "base_action_",
    "expert_action_",
    "residual_action_",
]

TARGET_RANGE = range(0, 7)  # inclusive [0, 6]


def delete_file(path):
    if DRY_RUN:
        print(f"[DRY RUN] Would delete: {path}")
    else:
        print(f"Deleting: {path}")
        os.remove(path)


def is_target_file(filename):
    """
    Check if filename matches:
    base_action_n.png / expert_action_n.png / residual_action_n.png
    where n in [0, 6]
    """
    if not filename.endswith(".png"):
        return False

    for prefix in TARGET_PREFIXES:
        if filename.startswith(prefix):
            try:
                n_part = filename[len(prefix):-4]  # strip prefix + ".png"
                n = int(n_part)
                return n in TARGET_RANGE
            except ValueError:
                return False

    return False


def main():
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        # Only act inside "viz" directories
        if os.path.basename(dirpath) != "viz":
            continue

        for fname in filenames:
            if is_target_file(fname):
                fpath = os.path.join(dirpath, fname)
                delete_file(fpath)


if __name__ == "__main__":
    main()
