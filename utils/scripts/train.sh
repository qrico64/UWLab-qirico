#!/bin/bash
set -euo pipefail

export SAVE_PATH=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar27/residual_o001s2r2_seed1
export RUN_NAME=r1_o1s2
export IS_EXPERT=0
export EPOCHS=1000
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
  export BASE_POLICY2=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/feb8/expert-ds_random5-receptive_x_geq_05-5layers_x4_relu/300-ckpt.pt
elif [ "$OUR_TASK" = "drawer" ]; then
  export BASE_POLICY=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar10/expert_drawer_y4_id/500-ckpt.pt
else
  echo "Unknown OUR_TASK=$OUR_TASK" >&2
  exit 1
fi

export CORRECTION_MODEL="${SAVE_PATH}/${EPOCHS}-ckpt.pt"
if [ "$IS_EXPERT" = "0" ]; then
  eval_job_id=$(
    sbatch --parsable \
      --job-name="e1$RUN_NAME" \
      --dependency=afterok:"$train_job_id" \
      --kill-on-invalid-dep=yes \
      --export=ALL,CORRECTION_MODEL="$CORRECTION_MODEL",OUR_TASK="$OUR_TASK",NUM_EVALS=5000,EVAL_RESET_MODE=xleq035,EVAL_MODE=default,BASE_POLICY="$BASE_POLICY" \
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
      --export=ALL,CORRECTION_MODEL="$CORRECTION_MODEL",OUR_TASK="$OUR_TASK",NUM_EVALS=5000,EVAL_RESET_MODE=xleq035,EVAL_MODE=obsnoise001,BASE_POLICY="none" \
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
      --export=ALL,SAVE_PATH="$SAVE_PATH",IS_EXPERT="$IS_EXPERT",EPOCHS="$EPOCHS",OUR_TASK="$OUR_TASK",BASE_POLICY="$BASE_POLICY",SAVE_PATH_NAME="finetune-feb22" \
      --output="$SAVE_PATH/log/ttt_%j_%x_out.txt" \
      --error="$SAVE_PATH/log/ttt_%j_%x_err.txt" \
      "$SAVE_PATH/ttt.slurm"
  )
  echo "Submitted ttt job:  $eval_job_id (depends on train job $train_job_id)"

  eval_job_id=$(
    sbatch --parsable \
      --job-name="t2$RUN_NAME" \
      --dependency=afterok:"$train_job_id" \
      --kill-on-invalid-dep=yes \
      --export=ALL,SAVE_PATH="$SAVE_PATH",IS_EXPERT="$IS_EXPERT",EPOCHS="$EPOCHS",OUR_TASK="$OUR_TASK",BASE_POLICY="$BASE_POLICY2",SAVE_PATH_NAME="finetune-feb8" \
      --output="$SAVE_PATH/log/ttt2_%j_%x_out.txt" \
      --error="$SAVE_PATH/log/ttt2_%j_%x_err.txt" \
      "$SAVE_PATH/ttt.slurm"
  )
  echo "Submitted ttt2 job:  $eval_job_id (depends on train job $train_job_id)"

elif [ "$IS_EXPERT" = "1" ]; then
  eval_job_id=$(
    sbatch --parsable \
      --job-name="e1$RUN_NAME" \
      --dependency=afterok:"$train_job_id" \
      --kill-on-invalid-dep=yes \
      --export=ALL,CORRECTION_MODEL="none",OUR_TASK="$OUR_TASK",NUM_EVALS=10000,EVAL_RESET_MODE=xleq035,EVAL_MODE=default,BASE_POLICY="$CORRECTION_MODEL" \
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
      --export=ALL,CORRECTION_MODEL="none",OUR_TASK="$OUR_TASK",NUM_EVALS=10000,EVAL_RESET_MODE=recxgeq05,EVAL_MODE=default,BASE_POLICY="$CORRECTION_MODEL" \
      --output="$SAVE_PATH/log/eval2_%j_%x_out.txt" \
      --error="$SAVE_PATH/log/eval2_%j_%x_err.txt" \
      "$SAVE_PATH/eval.slurm"
  )
  echo "Submitted eval2 job:  $eval_job_id (depends on train job $train_job_id)"

  eval_job_id=$(
    sbatch --parsable \
      --job-name="e3$RUN_NAME" \
      --dependency=afterok:"$train_job_id" \
      --kill-on-invalid-dep=yes \
      --export=ALL,CORRECTION_MODEL="none",OUR_TASK="$OUR_TASK",NUM_EVALS=10000,EVAL_RESET_MODE=none,EVAL_MODE=default,BASE_POLICY="$CORRECTION_MODEL" \
      --output="$SAVE_PATH/log/eval3_%j_%x_out.txt" \
      --error="$SAVE_PATH/log/eval3_%j_%x_err.txt" \
      "$SAVE_PATH/eval.slurm"
  )
  echo "Submitted eval3 job:  $eval_job_id (depends on train job $train_job_id)"
fi
