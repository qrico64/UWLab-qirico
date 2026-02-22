# python scripts/reinforcement_learning/rsl_rl/train2.py \
#     --lr 0.0003 \
#     --epochs 1000 \
#     --num_layers 4 \
#     --d_model 512 \
#     --dropout 0.1 \
#     --batch_size 256 \
#     --save_path experiments/feb20/residual_residual_obs0006_sys4_rand2_radius0003_6 \
#     --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb19/fourthtry_receptive_0006_sys4_rand2_recxgeq05/job-True-4.0-2.0-100000-60--0.006-0.0/cut-trajectories.pkl \
#     --train_mode closest-neighbors \
#     --closest_neighbors_radius 0.003 \
#     --warm_start 10 \
#     --train_percent 0.8 \
#     --infer_mode residual \
#     \
#     --receptive_xlow 0.5 \



python scripts/reinforcement_learning/rsl_rl/train2.py \
    --lr 0.0003 \
    --epochs 200 \
    --num_layers 4 \
    --d_model 512 \
    --dropout 0.3 \
    --batch_size 512 \
    --save_path experiments/feb21/expert_mlpblock_base_rand5_layernorm_dropout03_batch512 \
    --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb7/expertcol4/job-True-0.0-5.0-100000-60--0.0-0.0/cut-trajectories.pkl \
    --train_mode expert \
    --closest_neighbors_radius 0.001 \
    --warm_start 10 \
    --train_percent 0.8 \
    --infer_mode expert_new \
    \
    --head_arch_version mlpblock_v1 \
    --num_head_layers 5 \
    --d_model_head 2048 \
    --dropout_head 0.3 \
    \
    --receptive_xlow 0.5 \



# python scripts/reinforcement_learning/rsl_rl/train2.py \
#     --lr 0.0003 \
#     --epochs 300 \
#     --num_layers 4 \
#     --d_model 512 \
#     --dropout 0.1 \
#     --batch_size 256 \
#     --save_path experiments/feb9/expert-ds_random5-receptive_y_geq_0.2-5layers_x4_relu \
#     --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb4/expertcol5/job-True-0.0-5.0-100000-60--0.0-0.0/cut-trajectories.pkl \
#     --train_mode autoregressive \
#     --closest_neighbors_radius 0.001 \
#     --warm_start 10 \
#     --train_percent 0.8 \
#     --train_expert \
#     --finetuning_mode lora \
#     --model_from /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/feb8/expert-ds_random5-receptive_x_geq_05-5layers_x4_relu/300-ckpt.pt \
#     \
#     --use_new_head_arch \
#     --num_head_layers 5 \
#     --d_model_head 2048 \
#     \
#     --receptive_ylow 0.2 \

