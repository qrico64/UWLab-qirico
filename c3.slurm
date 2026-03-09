#!/bin/bash -l
#SBATCH --job-name=residual        # Job name
#SBATCH --output=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar5/residual_o10r2_lr1e_4/log/%j_%x_out.txt        # Output file (%j = job ID)
#SBATCH --error=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar5/residual_o10r2_lr1e_4/log/%j_%x_err.txt         # Error file
#SBATCH --time=20:00:00            # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=10          # CPUs per task
#SBATCH --gres=gpu:1               # GPUs per node (if needed)
#SBATCH --mem=100G                  # Memory per node
#SBATCH --partition=gpu-l40        # Partition (queue) name
#SBATCH --account=weirdlab         # Slurm account/project name

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
