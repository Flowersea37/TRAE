# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

# !!!!!!replace the model path before run

set -xeuo pipefail

project_name='GRPO-verl0.7.0release-Test'
exp_name='GRPO-Qwen3-8B-Base'

# adv_estimator=grpo
adv_estimator=grpo_tree_dynamic_advantage

use_kl_in_reward=False

clip_ratio_low=0.2
# clip_ratio_high=0.28
clip_ratio_high=0.4

max_prompt_length=$((1024 * 10))
max_response_length=$((1024 * 14))

# 这个参数可以train一下看影响大不大
loss_agg_mode="token-mean"

# enable_filter_groups=True

# [RUN]
train_prompt_bsz=256
train_prompt_mini_bsz=64
AGENT_NUM_WORKER=8

# [TEST RUN]
# train_prompt_bsz=64
# train_prompt_mini_bsz=64
# AGENT_NUM_WORKER=8

# [DEBUG]
# train_prompt_bsz=8
# train_prompt_mini_bsz=8
# AGENT_NUM_WORKER=8
# train_prompt_bsz=1
# train_prompt_mini_bsz=1
# AGENT_NUM_WORKER=1

# gen_prompt_bsz=$((train_prompt_bsz * 3))
gen_prompt_bsz=$((train_prompt_bsz))
# n_resp_per_prompt=16
n_resp_per_prompt=8
PPO_MICRO_BATCH_SIZE_PER_GPU=1


# Training Setting
TOTAL_EPOCHS=1
SAVE_EVERY_STEP=30
# Agent_loop name
AGENT_LOOP_NAME="trae_reflect_agent_while_style"

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:6397"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/config/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
# RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"Qwen3-8B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/saves_ckp/${project_name}/${exp_name}"}
# TRAIN_FILE=${TRAIN_FILE:-"/workspace/mnt/lxb_work/hf_dir/hf_dataset/Dapomath17k/data/dapo-math-17k.parquet"}
# TEST_FILE=${TEST_FILE:-"/workspace/mnt/lxb_work/hf_dir/hf_dataset/AIME-24/data/aime-2024.parquet"}
TRAIN_FILE="data/test_train_10k.parquet"
# train_data_path="/workspace/mnt/lxb_work/dlx_work/multi_verl/data/dapo17k/multi_reflect_train.parquet"
TEST_FILE="data/test_valid_10k.parquet"
# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
# use_dynamic_bsz=False
# actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
# infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
# offload=True
offload=False
gen_tp=1

ray stop --force
export RAY_TMPDIR="/tmp/ray_verl_dapo_test"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# RAY_DEBUG=legacy ray start --head --port=6398 --temp-dir=$RAY_TMPDIR --ray-debugger-external
RAY_DEBUG=legacy ray start --head --port=6398 --ray-debugger-external


python3 -m trae_verl.trainer.main_trae \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP_NAME} \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKER} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    reward_model.use_reward_loop=False \
    reward_model.reward_manager=trae \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_EVERY_STEP \
    trainer.test_freq=-1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="${CKPTS_DIR}" \
    +branch_per_node=2 \
    +max_turns=3 \
    "$@"
