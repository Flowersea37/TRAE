cd evaluation
REWARD_PATH=""

# 记录开始时间
START_TIME=$(date +%s)

# export CUDA_VISIBLE_DEVICES=0,5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray stop --force
# 计时 Ray 启动时间
ray start --head --num-gpus=8 --port=6342

export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL_PATH=""

# 计时 Python 程序执行时间
PYTHON_START_TIME=$(date +%s)

DATASET="math-500"
python eval_with_history.py \
    --policy_model $MODEL_PATH \
    --reward_model $REWARD_PATH \
    --dataset $DATASET \
    --mode "response" \
    --tensor_parallel_size 1 \
    --generation_out_seq_length 5000 \
    --times 3

DATASET="gsm8k"
python eval_with_history.py \
    --policy_model $MODEL_PATH \
    --reward_model $REWARD_PATH \
    --dataset $DATASET \
    --mode "response" \
    --tensor_parallel_size 1 \
    --generation_out_seq_length 5000 \
    --times 3

DATASET="olympiad"
python eval_with_history.py \
    --policy_model $MODEL_PATH \
    --reward_model $REWARD_PATH \
    --dataset $DATASET \
    --mode "response" \
    --tensor_parallel_size 1 \
    --generation_out_seq_length 5000 \
    --times 3

DATASET="minervamath"
python eval_with_history.py \
    --policy_model $MODEL_PATH \
    --reward_model $REWARD_PATH \
    --dataset $DATASET \
    --mode "response" \
    --tensor_parallel_size 1 \
    --generation_out_seq_length 5000 \
    --times 3

DATASET="amc23"
python eval_with_history.py \
    --policy_model $MODEL_PATH \
    --reward_model $REWARD_PATH \
    --dataset $DATASET \
    --mode "response" \
    --tensor_parallel_size 1 \
    --generation_out_seq_length 5000 \
    --times 3

DATASET="aime24"
python eval_with_history.py \
    --policy_model $MODEL_PATH \
    --reward_model $REWARD_PATH \
    --dataset $DATASET \
    --mode "response" \
    --tensor_parallel_size 1 \
    --generation_out_seq_length 5000 \
    --times 3

DATASET="aime25"
python eval_with_history.py \
    --policy_model $MODEL_PATH \
    --reward_model $REWARD_PATH \
    --dataset $DATASET \
    --mode "response" \
    --tensor_parallel_size 1 \
    --generation_out_seq_length 5000 \
    --times 3

PYTHON_END_TIME=$(date +%s)

# 计算总时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
PYTHON_TIME=$((PYTHON_END_TIME - PYTHON_START_TIME))

echo "==============================="
echo "计时结果:"
echo "Python 程序执行耗时: $PYTHON_TIME 秒"
echo "总耗时: $TOTAL_TIME 秒"
echo "==============================="

echo "Completed dataset: $dataset"
