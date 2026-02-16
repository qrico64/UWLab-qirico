eval "$(ssh-agent -s)"
ssh-add
conda activate env_uwlab

# export UW_BASE=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/docker
# apptainer instance stop uw-lab-1
# apptainer instance start --nv \
#     --bind /mmfs1/gscratch/stf/:/mmfs1/gscratch/stf/ \
#     --bind /gscratch/scrubbed/qirico/:/gscratch/scrubbed/qirico/ \
#     --bind /etc/pki:/etc/pki \
#     --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/ssl/certs/ca-certificates.crt \
#     --bind $UW_BASE/isaac-cache-kit:/isaac-sim/kit/cache \
#     --bind $UW_BASE/isaac-sim-data:/isaac-sim/kit/data \
#     --bind $UW_BASE/isaac-cache-ov:/root/.cache/ov \
#     --bind $UW_BASE/isaac-cache-pip:/root/.cache/pip \
#     --bind $UW_BASE/isaac-cache-gl:/root/.cache/nvidia/GLCache \
#     --bind $UW_BASE/isaac-cache-compute:/root/.nv/ComputeCache \
#     --bind $UW_BASE/logs:/workspace/uwlab/logs \
#     --bind $UW_BASE/outputs:/workspace/uwlab/outputs \
#     --bind $UW_BASE/data_storage:/workspace/uwlab/data_storage \
#     uw-lab-2_latest.sif \
#     uw-lab-1
# apptainer instance list
# apptainer shell --nv instance://uw-lab-1

# apptainer instance stop uw-lab-1
