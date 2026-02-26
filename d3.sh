python scripts/reinforcement_learning/rsl_rl/train2.py \
    --lr 0.0003 \
    --epochs 1000 \
    --num_layers 4 \
    --d_model 512 \
    --dropout 0.1 \
    --batch_size 256 \
    --save_path experiments/feb26/residual_s3r2_fulltraj_stateonly \
    --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb26/fourthtry_receptive_0_sys3_rand2_recxgeq05/job-True-3.0-2.0-100000-60--0.0-0.0/cut-trajectories.pkl \
    --train_mode full-traj \
    --closest_neighbors_radius 0.001 \
    --warm_start 10 \
    --train_percent 0.8 \
    --infer_mode res_scale_shift \
    \
    --head_arch_version mlpblock_v1 \
    --num_head_layers 5 \
    --d_model_head 2048 \
    --dropout_head 0.3 \
    \
    --mu_head_arch linear \
    --mu_size 512 \
    --mu_kl_factor 0.1 \
    \
    --current_head_arch none \
    --current_emb_size 512 \
    --current_kl_factor 0 \
    \
    --receptive_xlow 0.5 \



# python scripts/reinforcement_learning/rsl_rl/train2.py \
#     --lr 0.0003 \
#     --epochs 400 \
#     --num_layers 4 \
#     --d_model 512 \
#     --dropout 0.3 \
#     --batch_size 512 \
#     --save_path experiments/feb24/expert_mlpblockbase2_4layers_epoch400_klreg_d256_kl01 \
#     --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb7/expertcol4/job-True-0.0-5.0-100000-60--0.0-0.0/cut-trajectories.pkl \
#     --train_mode expert \
#     --closest_neighbors_radius 0.001 \
#     --warm_start 10 \
#     --train_percent 0.8 \
#     --infer_mode expert_new \
#     \
#     --head_arch_version mlpblock_v1 \
#     --num_head_layers 4 \
#     --d_model_head 2048 \
#     --dropout_head 0.3 \
#     \
#     --receptive_xlow 0.5 \

