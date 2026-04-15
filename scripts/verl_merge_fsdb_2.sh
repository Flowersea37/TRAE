export CUDA_VISIBLE_DEVICES=""

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /workspace/mnt/lxb_work/xgq_work/TRAE_upload/saves_ckp/GRPO-verl0.7.0release-Test/GRPO-Qwen3-8B-Base/global_step_117/actor \
    --target_dir /workspace/mnt/lxb_work/xgq_work/TRAE_upload/save_models/qwen3-8b-tree-best-config-10k-epoch1-dis05-tsp1-tokenloss-while-style-releasetest-step117
