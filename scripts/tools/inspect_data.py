import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def format_array_3dec(x: np.ndarray) -> str:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
        # Format float arrays to 3 decimal places
        return np.array2string(
            x,
            precision=3,
            floatmode='fixed',
            suppress_small=False
        )
    return str(x)


def save_point_distribution_image(x, out_path="dist.png", bins=400, dpi=200):
    """
    x: torch.Tensor, shape (N, 2) on CPU or GPU
    Saves a 2D histogram (density map) visualization to out_path.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    elif isinstance(x, list):
        x = np.array(x)

    fig, ax = plt.subplots()
    ax.hist2d(x[:, 0], x[:, 1], bins=bins)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("2D point distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    print(out_path)
    plt.close(fig)


def save_histogram(x, filename, bins=100):
    # convert to numpy
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    x = x.reshape(-1)

    plt.figure()
    plt.hist(x, bins=bins)
    plt.tight_layout()
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_actions_tsne(actions, n_components=2, filename="tsne_plot.png"):
    """
    Reduces 7D actions to 2D or 3D and saves a visualization.
    
    Args:
        actions (np.ndarray): Array of shape (N, 7).
        n_components (int): Dimensionality of the embedding (2 or 3).
        filename (str): The filename for the output image.
    """
    # t-SNE can be computationally expensive (O(N log N)). 
    # If your dataset is huge (e.g., >20k points), consider sub-sampling.
    if len(actions) > 10000:
        indices = np.random.choice(len(actions), 10000, replace=False)
        actions_to_fit = actions[indices]
    else:
        actions_to_fit = actions

    # Initialize t-SNE
    # Use init='pca' and learning_rate='auto' for better convergence
    tsne = TSNE(
        n_components=n_components, 
        init='pca', 
        learning_rate='auto', 
        random_state=42
    )
    actions_reduced = tsne.fit_transform(actions_to_fit)
    
    plt.clf() # Ensure a clean canvas
    
    if n_components == 2:
        plt.scatter(
            actions_reduced[:, 0], 
            actions_reduced[:, 1], 
            alpha=0.5, 
            s=1, 
            cmap='viridis'
        )
        plt.xlabel('$z_1$ (t-SNE)')
        plt.ylabel('$z_2$ (t-SNE)')
        plt.title(f'2D t-SNE Action Distribution (N={len(actions_to_fit)})')
        
    elif n_components == 3:
        ax = plt.subplot(111, projection='3d')
        # We use the 3rd component as a color map to help with depth perception
        scatter = ax.scatter(
            actions_reduced[:, 0], 
            actions_reduced[:, 1], 
            actions_reduced[:, 2], 
            c=actions_reduced[:, 2],
            alpha=0.5, 
            s=1, 
            cmap='viridis'
        )
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
        ax.set_zlabel('$z_3$')
        plt.title('3D t-SNE Action Distribution')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300) # High DPI for publication quality
    print(filename)


def main():
    FILENAME = "/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/cut-trajectories_jun16-True-2.0-0.0-40400.pkl"
    VIZ_DIR = f"viz/{os.path.basename(FILENAME)[:-4]}/"
    os.makedirs(VIZ_DIR, exist_ok=True)
    with open(FILENAME, "rb") as fi:
        trajs = pickle.load(fi)
    lengths = [traj['actions'].shape[0] for traj in trajs]
    save_histogram(lengths, VIZ_DIR + "lengths.png", bins=40)
    starting_positions = np.stack([traj['starting_position'][:2] for traj in trajs], axis=0)
    save_point_distribution_image(starting_positions, VIZ_DIR + "starting_positions.png")
    rewards = np.concatenate([traj['rewards'] for traj in trajs], axis=0)
    rewards = np.maximum(rewards, np.quantile(rewards, 0.01))
    save_histogram(rewards, VIZ_DIR + "rewards.png", bins=100)
    actions = np.concatenate([traj['actions'] for traj in trajs], axis=0)
    action_low = []
    action_high = []
    for i in range(7):
        actions_1dim = np.concatenate([traj['actions'][:,i] for traj in trajs], axis=0)
        action_low.append(round(np.quantile(actions_1dim, 0.1), 2))
        action_high.append(round(np.quantile(actions_1dim, 0.9), 2))
        print(f"Action dim {i}: 10% = {np.quantile(actions_1dim, 0.1)}, 90% = {np.quantile(actions_1dim, 0.9)}")
        actions_1dim = np.clip(actions_1dim, np.quantile(actions_1dim, 0.01), np.quantile(actions_1dim, 0.99))
        save_histogram(actions_1dim, VIZ_DIR + f"action_dim_{i}.png", bins=100)

        sys_noise_1dim = np.array([traj['sys_noise'][i] for traj in trajs])
        save_histogram(sys_noise_1dim, VIZ_DIR + f"sys_noise_{i}.png", bins=100)
    print(f"action low = {action_low}")
    print(f"action high = {action_high}")
    print(f"action mean = {actions.mean(axis=0).tolist()}")
    print(f"action std = {actions.std(axis=0).tolist()}")
    success = np.array([np.any(traj['rewards'] > 0.11) for traj in trajs])
    success_rate = np.mean(success)
    print(f"Success rate = {success_rate}")
    failed_sysnoise = np.stack([traj['sys_noise'] for traj in trajs], axis=0)[~success]
    pass

if __name__ == "__main__":
    main()
