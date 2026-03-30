import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
PATHS_TO_COMPARE = [
    "experiments/mar27/residual_o0015s2r2_seed1/finetune/viz/success_rate_across_seeds.txt",
    "experiments/mar29/residual_o0015s2r4_seed1/finetune-feb22/viz/success_rate_across_seeds.txt",
]
OUTPUT_FILE = "./comparison.png"

def load_processed_data(path):
    steps, means, stds = [], [], []
    with open(path, "r") as f:
        for line in f:
            s, m, std = line.strip().split()
            steps.append(int(s))
            means.append(float(m))
            stds.append(float(std))
    return np.array(steps), np.array(means), np.array(stds)

def main():
    plt.figure(figsize=(10, 6))

    for path in PATHS_TO_COMPARE:
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            continue
            
        # Use the parent directory name or a specific part of the path for the label
        label = path.split(os.sep)[-4] # Adjust index based on your folder structure
        
        steps, means, stds = load_processed_data(path)
        
        line, = plt.plot(steps, means, marker="o", markersize=4, linewidth=2, label=label)
        plt.fill_between(steps, means - stds, means + stds, color=line.get_color(), alpha=0.15)

    plt.xlabel("Checkpoint")
    plt.ylabel("Success Rate")
    plt.title("Comparison of Success Rates Across Experiments")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=200)
    print(f"Comparison plot saved to: {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()
