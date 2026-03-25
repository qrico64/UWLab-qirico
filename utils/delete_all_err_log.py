import os

# --- CONFIG ---
ROOT_DIR = "experiments"  # change or pass via CLI
DRY_RUN = False  # set to False to actually delete

# --- STATS ---
deleted_count = 0
freed_bytes = 0


def human_readable_size(num_bytes):
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def delete_file(path):
    global deleted_count, freed_bytes

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0

    if DRY_RUN:
        print(f"[DRY RUN] Would delete: {path} ({human_readable_size(size)})")
    else:
        print(f"Deleting: {path} ({human_readable_size(size)})")
        os.remove(path)

    deleted_count += 1
    freed_bytes += size


def is_err_log(filename):
    """Matches files like *_err.txt"""
    return filename.endswith("_err.txt")


def main():
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        # Only operate inside "log" directories
        if os.path.basename(dirpath) != "log":
            continue

        for fname in filenames:
            if is_err_log(fname):
                fpath = os.path.join(dirpath, fname)
                delete_file(fpath)

    # --- SUMMARY ---
    print("\n===== SUMMARY =====")
    print(f"Files matched: {deleted_count}")
    print(f"Total size freed: {human_readable_size(freed_bytes)}")
    if DRY_RUN:
        print("(Dry run mode - no files were actually deleted)")


if __name__ == "__main__":
    main()
