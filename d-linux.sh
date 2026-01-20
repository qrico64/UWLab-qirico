conda activate env_uwlab

python scripts/reinforcement_learning/rsl_rl/play_eval1.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --checkpoint peg_state_rl_expert.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --enable_cameras \
    --horizon 60 \
    --num_envs 1 \
    --correction_model /home/ricoqi/qirico/Meta-Learning-10-1/UWLab-qirico/experiments/jan18_fixnormalization_sysnoise-0.3-transformer-256-4-0.1/199-ckpt.pt
