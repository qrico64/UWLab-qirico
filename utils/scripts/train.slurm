#!/bin/bash -l
#SBATCH --time=24:00:00            # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=8          # CPUs per task
#SBATCH --gres=gpu:l40:1               # GPUs per node (if needed)
#SBATCH --mem=80G                  # Memory per node
#SBATCH --partition=ckpt-all        # Partition (queue) name
#SBATCH --account=weirdlab         # Slurm account/project name
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --open-mode=append

set -euo pipefail
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
conda activate env_uwlab

echo "Running with SLURM_SEED: $SLURM_SEED"
echo "$SAVE_PATH"
echo "$IS_EXPERT"
echo "$EPOCHS"
echo "$OUR_TASK"

export OBSNOISE_DS=/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar5/obs001r2_dataset_recxgeq05/job-True-0.0-2.0-100000-60--0.01-0.0/cut-trajectories.pkl
export SYSNOISE_DS=/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb26/fourthtry_receptive_0_sys3_rand2_recxgeq05/job-True-3.0-2.0-100000-60--0.0-0.0/cut-trajectories.pkl
export OBSNOISE_DS_NEW=/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar27/peg_recxgeq05_o0015s2r2/cut-trajectories.pkl

set +e
python scripts/reinforcement_learning/rsl_rl/train2.py \
    --lr 0.0001 \
    --epochs "$EPOCHS" \
    --num_layers 4 \
    --d_model 512 \
    --dropout 0.1 \
    --batch_size 256 \
    --save_path "$SAVE_PATH" \
    --dataset_path "$OBSNOISE_DS_NEW" \
    --train_mode perfect-coverage \
    --closest_neighbors_radius 0.001 \
    --warm_start 10 \
    --train_percent 0.8 \
    --infer_mode res_scale_shift \
    --state_type standard \
    --current_dim 45 \
    --our_task "$OUR_TASK" \
    --seed "$SLURM_SEED" \
    \
    --head_arch_version mlpblock_v1 \
    --num_head_layers 5 \
    --d_model_head 2048 \
    --dropout_head 0.3 \
    \
    --mu_head_arch 2layer \
    --mu_size 16 \
    --mu_kl_factor 0.001 \
    \
    --current_head_arch none \
    --current_emb_size 512 \
    --current_kl_factor 0 \
    \
    --combined_head_arch none \
    --combined_emb_size 1024 \
    --combined_kl_factor 0 \
    \
    --receptive_xlow 0.5

rc=$?
set -e

if [ "$rc" -eq 3 ]; then
    echo "Training checkpointed for restart; requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

exit "$rc"
