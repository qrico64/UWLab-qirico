import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
OUTPUT_FILE = "./simple_plot.png"

def main():
    # Hard-coded data
    x = np.array([1, 3, 5])
    # TITLE = "Drawer"
    # y1 = np.array([0.839, 0.725, 0.725])
    # y2 = np.array([0.094, 0.596, 0.595])
    # TITLE = "Drawer (training y >= 0.15)"
    # y1 = np.array([0.820, 0.720, 0.717])
    # y2 = np.array([0.118, 0.312, 0.414])

    plt.figure(figsize=(10, 6))

    # First curve (id)
    plt.plot(x, y1, marker="o", markersize=4, linewidth=2, label="id")

    # Second curve (ood)
    plt.plot(x, y2, marker="o", markersize=4, linewidth=2, label="ood")

    plt.xlabel("History Length")
    plt.ylabel("Success Rate")
    plt.title(TITLE)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=200)
    print(f"Plot saved to: {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()
