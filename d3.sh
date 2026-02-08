# python scripts/reinforcement_learning/rsl_rl/train2.py \
#     --lr 0.0003 \
#     --epochs 300 \
#     --num_layers 4 \
#     --d_model 512 \
#     --dropout 0.1 \
#     --batch_size 256 \
#     --save_path experiments/feb2/obsnoise_0.01_normboth_0.8dataset \
#     --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/jan29/job-True-0.0-0.0-60000-60--0.01-0.0/cut-trajectories.pkl \
#     --train_mode closest-neighbors \
#     --closest_neighbors_radius 0.001 \
#     --warm_start 10 \
#     --train_percent 0.8 \



python scripts/reinforcement_learning/rsl_rl/train2.py \
    --lr 0.0003 \
    --epochs 300 \
    --num_layers 4 \
    --d_model 512 \
    --dropout 0.1 \
    --batch_size 256 \
    --save_path experiments/feb8/expert-ds_random5-receptive_x_geq_05-5layers_x4_relu \
    --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb7/expertcol4/job-True-0.0-5.0-100000-60--0.0-0.0/cut-trajectories.pkl \
    --train_mode autoregressive \
    --closest_neighbors_radius 0.001 \
    --warm_start 10 \
    --train_percent 0.8 \
    --train_expert \
    \
    --use_new_head_arch \
    --num_head_layers 5 \
    --d_model_head 2048 \
    \
    --receptive_xlow 0.5 \



# python scripts/reinforcement_learning/rsl_rl/train_mlp.py \
#     --lr 0.0003 \
#     --epochs 300 \
#     --action_low -0.999 \
#     --action_high 0.999 \
#     --num_layers 4 \
#     --d_model 1536 \
#     --dropout 0.1 \
#     --batch_size 256 \
#     --save_path experiments/jan29/what_if_we_just_had_larger_gaussian \
#     --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/cut-trajectories_jan17-True-3.0-0.0-60000.pkl \
#     --model_type gaussian \
