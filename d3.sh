python scripts/reinforcement_learning/rsl_rl/train2.py \
    --lr 0.0003 \
    --epochs 300 \
    --action_low -0.999 \
    --action_high 0.999 \
    --num_layers 4 \
    --d_model 256 \
    --dropout 0.1 \
    --save_path experiments/sysnoise-0.2-transformer-256-4-0.1
