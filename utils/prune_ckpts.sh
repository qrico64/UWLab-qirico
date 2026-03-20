export SAVE_PATH=experiments/mar17/residual_o003s4r2_lr1e_4_stateonly/finetune-xleq035-f22

# python utils/prune_ckpts.py $SAVE_PATH
python utils/replot_success_rates.py $SAVE_PATH