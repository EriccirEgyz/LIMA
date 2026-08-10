#!/usr/bin/env bash
# GRPO | Qwen3 Small Models | SGLang | NVIDIA GPUs
# EnvFactory

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

########################### user-adjustable ###########################
INFER_BACKEND=sglang

PROJECT_DIR="$PWD"
MODEL_PATH=/data/user/mxubh/LLaMAFactory/models/qwen3-4b/env_factory_sft
NNODES=1
NGPUS_PER_NODE=8

train_files=data/env_factory_rl.json
val_files=data/env_factory_rl_val.json

train_batch_size=256
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=2
max_prompt_length=16384
max_response_length=8192

rollout_tp=1
rollout_gpu_mem_util=0.8
rollout_n=8
rollout_temperature=0.7
rollout_log_prob_micro_batch_size_per_gpu=8

total_epochs=10
save_freq=20
test_freq=5

project_name=EnvFactory
experiment_name=${project_name}-RL
########################### end user-adjustable ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

REWARD=(
    reward.custom_reward_function.path=EnvFactory/reward/tool_reward_fcn.py
    reward.custom_reward_function.name=compute_tool_reward
    +data.reward_scheduler.mode=fix
    +data.reward_scheduler.start_ratio=0.9
    +data.reward_scheduler.end_ratio=0.1
    +data.reward_scheduler.default_ratio=0.5
)

DATA=(
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.rollout.trace.backend=tensorboard
    actor_rollout_ref.rollout.trace.token2text=True
    actor_rollout_ref.rollout.multi_turn.format=qwen3
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.validation_data_dir=log_validation/${experiment_name}
    trainer.rollout_data_dir=log_rollout/${experiment_name}
)

python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"

# trainer.resume_mode=resume_path \
# trainer.resume_from_path=/data/user/mxubh/verl/checkpoints/MCPFactory/Qwen3-1.7B-n8-sglang-no_kl-grpo-0.5-1e-6-SFT-efficient-20260320-1118/global_step_200 \
# gpu_memory_utilization: 8b: 0.6; 4b: 0.85 1.7b: 0.9