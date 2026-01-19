python scripts/reinforcement_learning/rsl_rl/train2.py \
    --lr 0.0003 \
    --epochs 300 \
    --action_low -0.999 \
    --action_high 0.999 \
    --num_layers 4 \
    --d_model 256 \
    --dropout 0.1 \
    --save_path experiments/jan18_fixnormalization_sysnoise-0.4-transformer-256-4-0.1 \
    --dataset_path /mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/cut-trajectories_jan17-True-4.0-0.0-60000.pkl
