#!/bin/bash -l
#SBATCH --job-name=r_dro1k3        # Job name
#SBATCH --output=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar11/residual_drawer_o001r2_lr1e_4_perfect_cov_kl_mu_1e_3/log/%j_%x_out.txt        # Output file (%j = job ID)
#SBATCH --error=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar11/residual_drawer_o001r2_lr1e_4_perfect_cov_kl_mu_1e_3/log/%j_%x_err.txt         # Error file
#SBATCH --time=20:00:00            # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=2          # CPUs per task
#SBATCH --gres=gpu:1               # GPUs per node (if needed)
#SBATCH --mem=38G                  # Memory per node
#SBATCH --partition=gpu-l40s        # Partition (queue) name
#SBATCH --account=stf         # Slurm account/project name

# Load environment
cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico

# Checks
echo
echo "Node: $(hostname)"
which python
python -V
python -c "import sys, pprint; pprint.pprint(sys.path[:5])"
echo
echo


# Run your program
source a.sh
source d3.sh
