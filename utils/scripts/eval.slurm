#!/bin/bash -l
#SBATCH --time=24:00:00            # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=2          # CPUs per task
#SBATCH --gres=gpu:a40:1               # GPUs per node (if needed)
#SBATCH --mem=80G                  # Memory per node
#SBATCH --partition=ckpt-all        # Partition (queue) name
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
conda activate env_uwlab

echo $OUR_TASK
echo $NUM_EVALS
echo $EVAL_MODE
echo $EVAL_RESET_MODE
echo $BASE_POLICY
echo $CORRECTION_MODEL

export UW_BASE=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/docker

export APPTAINERENV_ISAACSIM_PATH=/isaac-sim/
export APPTAINERENV_OMNI_USER_DATA_PATH=/tmp/qirico/ov/data
export APPTAINERENV_OMNI_CACHE_PATH=/tmp/qirico/ov/cache
export APPTAINERENV_TERM=xterm-256color
mkdir -p $APPTAINERENV_OMNI_USER_DATA_PATH $APPTAINERENV_OMNI_CACHE_PATH

export JOBTMP=/tmp/${USER}_tmp_${SLURM_JOB_ID:-manual}_$$
mkdir -p "$JOBTMP"
chmod 700 "$JOBTMP"

apptainer exec --nv \
  --env OUR_TASK="$OUR_TASK" \
  --env NUM_EVALS="$NUM_EVALS" \
  --env EVAL_MODE="$EVAL_MODE" \
  --env EVAL_RESET_MODE="$EVAL_RESET_MODE" \
  --env BASE_POLICY="$BASE_POLICY" \
  --env CORRECTION_MODEL="$CORRECTION_MODEL" \
  --bind /mmfs1/gscratch/stf/:/mmfs1/gscratch/stf/ \
  --bind /gscratch/scrubbed/qirico/:/gscratch/scrubbed/qirico/ \
  --bind /etc/pki:/etc/pki \
  --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/ssl/certs/ca-certificates.crt \
  --bind $UW_BASE/isaac-cache-kit:/isaac-sim/kit/cache \
  --bind $UW_BASE/isaac-sim-data:/isaac-sim/kit/data \
  --bind $UW_BASE/isaac-cache-ov:/root/.cache/ov \
  --bind $UW_BASE/isaac-cache-pip:/root/.cache/pip \
  --bind $UW_BASE/isaac-cache-gl:/root/.cache/nvidia/GLCache \
  --bind $UW_BASE/isaac-cache-compute:/root/.nv/ComputeCache \
  --bind $UW_BASE/logs:/workspace/uwlab/logs \
  --bind $UW_BASE/outputs:/workspace/uwlab/outputs \
  --bind $UW_BASE/data_storage:/workspace/uwlab/data_storage \
  --bind "$JOBTMP:/tmp" \
  --bind $(pwd):/workspace/uwlab \
  uw-lab-2_latest.sif \
  bash -lc '

HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
  --our_task $OUR_TASK \
  --headless \
  --num_envs 100 \
  --num_evals $NUM_EVALS \
  --base_policy $BASE_POLICY \
  --correction_model $CORRECTION_MODEL \
  --reset_mode $EVAL_RESET_MODE \
  --eval_mode $EVAL_MODE \
'
