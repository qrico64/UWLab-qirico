export ISAACSIM_PATH="/isaac-sim/"
export OMNI_USER_DATA_PATH="/tmp/qirico/ov/data"
export OMNI_CACHE_PATH="/tmp/qirico/ov/cache"
mkdir -p $OMNI_USER_DATA_PATH $OMNI_CACHE_PATH
export TERM=xterm-256color

# /isaac-sim/isaac-sim.sh --reset-user --no-window --vulkan-device-index 0
# /isaac-sim/python.sh -V

# /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
#     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
#     --checkpoint peg_state_rl_expert.pt \
#     env.scene.insertive_object=peg \
#     env.scene.receptive_object=peghole \
#     --headless \
#     --num_trajectories 100 \
#     --num_envs 100

/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --checkpoint peg_state_rl_expert.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --num_envs 1 \
    --enable_cameras \

# /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play2.py \
#     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
#     --checkpoint peg_state_rl_expert.pt \
#     env.scene.insertive_object=peg \
#     env.scene.receptive_object=peghole \
#     --headless \
#     --num_envs 1 \
#     --num_trajectories 100

# /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
#     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
#     --checkpoint peg_state_rl_expert.pt \
#     env.scene.insertive_object=peg \
#     env.scene.receptive_object=peghole \
#     --headless \
#     --num_trajectories 10000 \
#     --num_envs 1000
