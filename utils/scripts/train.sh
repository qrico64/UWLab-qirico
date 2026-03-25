#!/bin/bash
set -euo pipefail

export SAVE_PATH=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar25/test_residual
export RUN_NAME=r_test2
export IS_EXPERT=0
export EPOCHS=20
export OUR_TASK=peg

mkdir -p "$SAVE_PATH"
mkdir -p "$SAVE_PATH/log"
cp utils/scripts/train.sh "$SAVE_PATH/"
cp utils/scripts/train.slurm "$SAVE_PATH/"
cp utils/scripts/eval.slurm "$SAVE_PATH/"
cp utils/scripts/ttt.slurm "$SAVE_PATH/"
cp scripts/reinforcement_learning/rsl_rl/train2.py "$SAVE_PATH/"
cp scripts/reinforcement_learning/rsl_rl/train_lib.py "$SAVE_PATH/"
cp scripts/reinforcement_learning/rsl_rl/play_eval1.py "$SAVE_PATH/"

train_job_id=$(
  sbatch --parsable \
    --job-name="$RUN_NAME" \
    --export=ALL,SAVE_PATH="$SAVE_PATH",IS_EXPERT="$IS_EXPERT",EPOCHS="$EPOCHS",OUR_TASK="$OUR_TASK" \
    --output="$SAVE_PATH/log/train_%j_%x_out.txt" \
    --error="$SAVE_PATH/log/train_%j_%x_err.txt" \
    "$SAVE_PATH/train.slurm"
)

echo "Submitted train job: $train_job_id"

if [ "$OUR_TASK" = "peg" ]; then
  export BASE_POLICY=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/feb22/expert_mlpblockbase2_4layers_epoch400/400-ckpt.pt
elif [ "$OUR_TASK" = "drawer" ]; then
  export BASE_POLICY=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar10/expert_drawer_y4_id/500-ckpt.pt
else
  echo "Unknown OUR_TASK=$OUR_TASK" >&2
  exit 1
fi

if [ "$IS_EXPERT" = "0" ]; then
  eval_job_id=$(
    sbatch --parsable \
      --job-name="e1$RUN_NAME" \
      --dependency=afterok:"$train_job_id" \
      --kill-on-invalid-dep=yes \
      --export=ALL,SAVE_PATH="$SAVE_PATH",IS_EXPERT="$IS_EXPERT",EPOCHS="$EPOCHS",OUR_TASK="$OUR_TASK",NUM_EVALS=5000,EVAL_RESET_MODE=xleq035,EVAL_MODE=default,BASE_POLICY="$BASE_POLICY" \
      --output="$SAVE_PATH/log/eval1_%j_%x_out.txt" \
      --error="$SAVE_PATH/log/eval1_%j_%x_err.txt" \
      "$SAVE_PATH/eval.slurm"
  )
  echo "Submitted eval1 job:  $eval_job_id (depends on train job $train_job_id)"

  eval_job_id=$(
    sbatch --parsable \
      --job-name="e2$RUN_NAME" \
      --dependency=afterok:"$train_job_id" \
      --kill-on-invalid-dep=yes \
      --export=ALL,SAVE_PATH="$SAVE_PATH",IS_EXPERT="$IS_EXPERT",EPOCHS="$EPOCHS",OUR_TASK="$OUR_TASK",NUM_EVALS=5000,EVAL_RESET_MODE=xleq035,EVAL_MODE=obsnoise001,BASE_POLICY="none" \
      --output="$SAVE_PATH/log/eval2_%j_%x_out.txt" \
      --error="$SAVE_PATH/log/eval2_%j_%x_err.txt" \
      "$SAVE_PATH/eval.slurm"
  )
  echo "Submitted eval2 job:  $eval_job_id (depends on train job $train_job_id)"

  eval_job_id=$(
    sbatch --parsable \
      --job-name="t$RUN_NAME" \
      --dependency=afterok:"$train_job_id" \
      --kill-on-invalid-dep=yes \
      --export=ALL,SAVE_PATH="$SAVE_PATH",IS_EXPERT="$IS_EXPERT",EPOCHS="$EPOCHS",OUR_TASK="$OUR_TASK",BASE_POLICY="$BASE_POLICY" \
      --output="$SAVE_PATH/log/ttt_%j_%x_out.txt" \
      --error="$SAVE_PATH/log/ttt_%j_%x_err.txt" \
      "$SAVE_PATH/ttt.slurm"
  )
  echo "Submitted ttt job:  $eval_job_id (depends on train job $train_job_id)"
fi
