import matplotlib.pyplot as plt

# Data
x = [0, 1e-5, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 6e-1]
y = [
    0.462327241897583,
    0.45248475670814514,
    0.5603026151657104,
    0.5227078199386597,
    0.5643044710159302,
    0.571490466594696,
    0.5497177839279175,
    0.5563002824783325,
]
x2 = [0, 1e-5, 1e-3, 1e-2, 1e-1]
y2 = [
    0.46623656153678894,
    0.4828646779060364,
    0.5182169079780579,
    0.571860134601593,
    0.5352863073348999,
]

# Figure (higher DPI for sharper rendering)
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(x, y, marker="o", label="Mu dimension 64")
plt.plot(x2, y2, marker="o", label="Mu dimension 512")
plt.hlines(0.5385, xmin=0, xmax=6e-1, colors="gray", linestyles="dashed", label="Expert (First-try) success rate")

plt.xlabel("KL divergence loss weight")
plt.ylabel("Second-try eval success rate")
plt.title("Effect of Information Bottleneck on Residual Model")
plt.legend()

# symlog keeps x=0 visible while still spreading small values
plt.xscale("symlog", linthresh=1e-6)

plt.tight_layout()
plt.savefig("kl_effect.png")
plt.show()
plt.close()
