import os
import re

ROOT_DIR = "experiments/mar27"

CKPT_PATTERN = re.compile(r"^(\d+)-ckpt\.pt$")
EVAL_VIZ_PATTERN = re.compile(r"^(\d+)-ckpt-eval_viz$")

DRY_RUN = True  # set to False to actually delete

total_deleted_size = 0  # NEW: Track total size


def delete_file(path):
    global total_deleted_size
    if not os.path.exists(path): return  # Simple check for latest.pt
    
    size = os.path.getsize(path)
    if DRY_RUN:
        print(f"[DRY RUN] Would delete: {path} ({size / 1e6:.2f} MB)")
    else:
        print(f"Deleting: {path} ({size / 1e6:.2f} MB)")
        os.remove(path)
    total_deleted_size += size


def process_finetune_directory(dirpath, filenames, dirnames):
    ckpts = []
    eval_viz_steps = set()

    for fname in filenames:
        m = CKPT_PATTERN.match(fname)
        if m:
            step = int(m.group(1))
            if step >= 900 or (step >= 90 and step <= 150):
                continue
            ckpts.append((step, fname))

    for dname in dirnames:
        m = EVAL_VIZ_PATTERN.match(dname)
        if m:
            step = int(m.group(1))
            eval_viz_steps.add(step)

    for step, fname in ckpts:
        if step in eval_viz_steps:
            fpath = os.path.join(dirpath, fname)
            delete_file(fpath)

            # --- NEW: also delete latest.pt ---
            latest_path = os.path.join(dirpath, "latest.pt")
            delete_file(latest_path)
        else:
            print(f"Keeping (no eval_viz): {os.path.join(dirpath, fname)}")


def process_regular_directory(dirpath, filenames):
    # --- NEW: also delete latest.pt ---
    latest_path = os.path.join(dirpath, "latest.pt")
    delete_file(latest_path)

    ckpts = []
    for fname in filenames:
        m = CKPT_PATTERN.match(fname)
        if m:
            step = int(m.group(1))
            ckpts.append((step, fname))

    if len(ckpts) <= 1:
        return

    ckpts.sort(key=lambda x: x[0])
    to_delete = ckpts[:-1]
    kept = ckpts[-1][1]

    print(f"Keeping final checkpoint: {os.path.join(dirpath, kept)}")
    for _, fname in to_delete:
        fpath = os.path.join(dirpath, fname)
        delete_file(fpath)


def main():
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        dirname = os.path.basename(dirpath).lower()

        if "finetune" in dirname:
            process_finetune_directory(dirpath, filenames, dirnames)
        else:
            process_regular_directory(dirpath, filenames)
            
    print(f"\nTotal size {'to be deleted' if DRY_RUN else 'deleted'}: {total_deleted_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()