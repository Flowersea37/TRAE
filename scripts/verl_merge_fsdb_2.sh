export CUDA_VISIBLE_DEVICES=""
# !!!!!!replace the model path before run

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir global_step_117/actor \
    --target_dir ./qwen3-8b-tree-best-config-10k-epoch1-dis05-tsp1-tokenloss-while-style-releasetest-step117
