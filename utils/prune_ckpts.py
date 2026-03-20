import os
import shutil
import argparse
from datetime import datetime

# --- ARGPARSE ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "target_dir",
    type=str,
    help="Target directory containing checkpoints",
)
args = parser.parse_args()

TARGET_DIR = args.target_dir

# --- CONFIG ---
CUTOFF = datetime(2026, 3, 19, 17, 0, 0)  # Mar 19, 17:00
MAX_ALLOWED_DELTA = 3600  # 1 hour in seconds

cutoff_ts = CUTOFF.timestamp()

# --- ASSERTION ---
for fname in os.listdir(TARGET_DIR):
    if fname.endswith("-ckpt.pt"):
        ckpt_path = os.path.join(TARGET_DIR, fname)

        mtime = os.path.getmtime(ckpt_path)

        # NOTE: keeping your original condition exactly as written
        if abs(mtime - cutoff_ts) < MAX_ALLOWED_DELTA:
            raise AssertionError(
                f"Timestamp drift too large for {ckpt_path}: "
                f"mtime={datetime.fromtimestamp(mtime)}, cutoff={CUTOFF}"
            )

# --- MAIN LOGIC ---
for fname in os.listdir(TARGET_DIR):
    if fname.endswith("-ckpt.pt"):
        ckpt_path = os.path.join(TARGET_DIR, fname)

        mtime = os.path.getmtime(ckpt_path)

        if mtime < cutoff_ts:
            prefix = fname[:-len("-ckpt.pt")]

            print(f"Removing {ckpt_path}")
            os.remove(ckpt_path)

            eval_dir = os.path.join(TARGET_DIR, f"{prefix}-ckpt-eval_viz")
            if os.path.isdir(eval_dir):
                print(f"Removing {eval_dir}")
                shutil.rmtree(eval_dir)
