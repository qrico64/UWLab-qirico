export SAVE_PATH=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar24/residual_o005s5r2_lr1e_4_perfect_cov_kl_mu_1e_3_d16_seed3

export OBSNOISE_DS=/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar5/obs001r2_dataset_recxgeq05/job-True-0.0-2.0-100000-60--0.01-0.0/cut-trajectories.pkl
export SYSNOISE_DS=/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb26/fourthtry_receptive_0_sys3_rand2_recxgeq05/job-True-3.0-2.0-100000-60--0.0-0.0/cut-trajectories.pkl
export OBSNOISE_DS_NEW=/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar10/peg_recxgeq05_id_obs003_sys4_r2/cut-trajectories.pkl

mkdir -p $SAVE_PATH
cp d3.sh $SAVE_PATH/
cp scripts/reinforcement_learning/rsl_rl/train2.py $SAVE_PATH/
cp scripts/reinforcement_learning/rsl_rl/train_lib.py $SAVE_PATH/
cp scripts/reinforcement_learning/rsl_rl/play_eval1.py $SAVE_PATH/

export IS_EXPERT=0
export EPOCHS=1000
export OUR_TASK=peg
python scripts/reinforcement_learning/rsl_rl/train2.py \
    --lr 0.0001 \
    --epochs $EPOCHS \
    --num_layers 4 \
    --d_model 512 \
    --dropout 0.1 \
    --batch_size 256 \
    --save_path $SAVE_PATH \
    --dataset_path $OBSNOISE_DS_NEW \
    --train_mode perfect-coverage \
    --closest_neighbors_radius 0.001 \
    --warm_start 10 \
    --train_percent 0.8 \
    --infer_mode res_scale_shift \
    --state_type standard \
    --current_dim 45 \
    --our_task $OUR_TASK \
    --seed 42 \
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
    --receptive_xlow 0.5 \


# export IS_EXPERT=1
# export EPOCHS=500
# export OUR_TASK=peg
# python scripts/reinforcement_learning/rsl_rl/train2.py \
#     --lr 0.0003 \
#     --epochs $EPOCHS \
#     --num_layers 4 \
#     --d_model 512 \
#     --dropout 0.3 \
#     --batch_size 512 \
#     --save_path $SAVE_PATH \
#     --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar15/drawer_y5_id_h110/cut-trajectories.pkl \
#     --train_mode expert \
#     --closest_neighbors_radius 0.001 \
#     --warm_start 10 \
#     --train_percent 0.8 \
#     --val_percent 0.2 \
#     --infer_mode expert_new \
#     --state_type standard \
#     --current_dim 45 \
#     \
#     --head_arch_version mlpblock_v1 \
#     --num_head_layers 5 \
#     --d_model_head 2048 \
#     --dropout_head 0.3 \



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
  --env SAVE_PATH="$SAVE_PATH" \
  --env EPOCHS="$EPOCHS" \
  --env OUR_TASK="$OUR_TASK" \
  --env IS_EXPERT="$IS_EXPERT" \
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

if [ "$OUR_TASK" = "peg" ]; then
  export BASE_POLICY=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/feb22/expert_mlpblockbase2_4layers_epoch400/400-ckpt.pt
elif [ "$OUR_TASK" = "drawer" ]; then
  export BASE_POLICY=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/mar10/expert_drawer_y4_id/500-ckpt.pt
fi
export CORRECTION_MODEL=${SAVE_PATH}/${EPOCHS}-ckpt.pt
export SAVE_PATH_NEW=${SAVE_PATH}/finetune

if [ "$IS_EXPERT" = "0" ]; then
  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 5000 \
    --base_policy $BASE_POLICY \
    --correction_model $CORRECTION_MODEL \
    --reset_mode xleq035 \
    --eval_mode default

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 5000 \
    --correction_model $CORRECTION_MODEL \
    --reset_mode xleq035 \
    --eval_mode obsnoise001
  
  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval2.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 10 \
    --num_evals 1000 \
    --finetune_mode residual \
    --base_policy $BASE_POLICY \
    --correction_model $CORRECTION_MODEL \
    --save_path $SAVE_PATH_NEW \
    --utd_ratio 1.0 \
    --finetune_arch lora \
    --lr 3e-4 \
    --reset_mode xleq035

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 2000 \
    --correction_model $SAVE_PATH_NEW \
    --reset_mode xleq035 \
    --eval_mode default

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 20000 \
    --correction_model $CORRECTION_MODEL \
    --reset_mode recxgeq05 \
    --eval_mode obsnoise001

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 5000 \
    --correction_model $CORRECTION_MODEL \
    --reset_mode xleq035 \
    --eval_mode default

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 5000 \
    --correction_model $CORRECTION_MODEL \
    --reset_mode recxgeq05 \
    --eval_mode default

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 20000 \
    --base_policy $BASE_POLICY \
    --correction_model $CORRECTION_MODEL \
    --reset_mode recxgeq05 \
    --eval_mode default

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 20000 \
    --correction_model $CORRECTION_MODEL \
    --reset_mode none \
    --eval_mode obsnoise001

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 20000 \
    --base_policy $BASE_POLICY \
    --correction_model $CORRECTION_MODEL \
    --reset_mode none \
    --eval_mode default
  
elif [ "$IS_EXPERT" = "1" ]; then

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 10000 \
    --correction_model $CORRECTION_MODEL \
    --reset_mode y4_id \
    --eval_mode default

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 10000 \
    --correction_model $CORRECTION_MODEL \
    --reset_mode y4_ood \
    --eval_mode default

  HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --our_task $OUR_TASK \
    --headless \
    --num_envs 100 \
    --num_evals 10000 \
    --correction_model $CORRECTION_MODEL \
    --reset_mode none \
    --eval_mode default
fi
'



