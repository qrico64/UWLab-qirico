import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
PATHS_TO_COMPARE = [
    "experiments/apr6/residual_o0015s4r2_ygeq015_kl1e_3_seed4/finetune-id_expert/viz/success_rate_across_seeds.txt",
    "experiments/apr6/residual_o0015s4r2_ygeq015_kl1e_3_seed4/finetune-ood_expert/viz/success_rate_across_seeds.txt",
    "experiments/apr6/residual_o0015s4r2_ygeq015_kl1e_3_seed4/finetune-apr3/viz/success_rate_across_seeds.txt",
]
OUTPUT_FILE = "./comparison.png"

def load_processed_data(path):
    steps, means, stds = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 2:
                s, m = line
                std = 0
            elif len(line) == 3:
                s, m, std = line
            steps.append(int(s))
            means.append(float(m))
            stds.append(float(std))

    steps = np.array(steps)
    means = np.array(means)
    stds = np.array(stds)

    # --- Added sorting by step ---
    order = np.argsort(steps)
    steps = steps[order]
    means = means[order]
    stds = stds[order]

    return steps, means, stds

def main():
    plt.figure(figsize=(10, 6))

    for path in PATHS_TO_COMPARE:
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            continue
            
        label = path.split(os.sep)[-3]
        
        steps, means, stds = load_processed_data(path)
        
        line, = plt.plot(steps, means, marker="o", markersize=4, linewidth=2, label=label)
        plt.fill_between(steps, means - stds, means + stds, color=line.get_color(), alpha=0.15)

    plt.axhline(0.420, linestyle="--", linewidth=2, color="red", label="base policy ood")
    plt.axhline(0.876, linestyle="--", linewidth=2, color="green", label="base policy id")

    plt.xlabel("Checkpoint")
    plt.ylabel("Success Rate")
    plt.title("Finetuning using Residual Model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=200)
    print(f"Comparison plot saved to: {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()
