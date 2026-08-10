#!/usr/bin/env bash
# GRPO RL for EnvFactory-1.7B reproduction | SGLang rollout | 1×H100 (80GB)
# Faithful port of the verl fork's run_qwen3_8b.sh (release/v0.6.1, commit 99f4e378).
# All algorithmic / reward / batch hyperparameters are kept IDENTICAL to run_qwen3_8b.sh.
# Deviations are only what a single GPU + unreleased val file force (see table below).
#
#   PARAM                        run_qwen3_8b.sh     HERE        reason
#   NGPUS_PER_NODE               8                   1           single GPU
#   rollout.gpu_memory_utilization 0.8               0.5         1-GPU colocation (OOM knob #1)
#   rollout.load_format          dummy               auto        1-GPU: dummy+async-agent-loop weight
#                                                                   sync yields garbage; auto loads real ckpt
#   rollout.free_cache_engine    True                False       1-GPU: init-sleep races sglang HTTP readiness
#   val_files                    env_factory_rl_val.json  env_factory_rl.json  val not released (Option B)
#   trainer.test_freq            5                   -1          val not released (Option B)
#   trainer.val_before_train     True                False       val not released (Option B)
#   model.path                   (theirs)            rl_init dir SFT epoch-1 ckpt + canonical tokenizer
#   everything else (256/64/2, n=8, lr=1e-6, kl=0.001,            identical
#     10 epochs, scheduler fix/0.5, grpo, fused_kernels ...)
#
# Environment is fully aligned with the verl 0.6 Docker: flash_attn==2.7.4.post1 is built
# from source against torch 2.8.0+cu128 (see setup_rl_env.sh), so use_fused_kernels=True
# runs aligned with run_qwen3_8b.sh.
#
# Smoke-test status (validated): full pipeline runs end-to-end (rollout -> reward -> PPO ->
# save). load_format=auto makes sglang generate coherent multi-turn tool calls (verified in
# the dump). Watch at scale: (1) gpu_memory_utilization if OOM; (2) CPU RAM ~85GB -> the
# DataLoader worker gets OOM-killed at end-of-run (harmless for completed steps, but monitor
# for long runs); (3) reward signal -- epoch-1 model gives 0 reward on a tiny smoke sample,
# confirm nonzero reward appears in the first steps of the full run.
#
# Smoke test (once a GPU is free), passthrough overrides (>=8 samples so prompt filtering
# doesn't empty the batch):
#   bash run_qwen3_1.7b_rl.sh data.train_max_samples=8 data.val_max_samples=8 \
#       data.train_batch_size=4 actor_rollout_ref.actor.ppo_mini_batch_size=4 \
#       trainer.total_training_steps=1 actor_rollout_ref.rollout.n=2
set -eo pipefail   # no -u: conda's cuda-nvcc activate.d references unbound vars

VERL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VERL_DIR"

# --- environment ---
source /workspace/shenchengyu/miniconda3/etc/profile.d/conda.sh
conda activate envfactory-rl
export CUDA_HOME="$CONDA_PREFIX"        # conda-forge cuda-nvcc 12.8 (sglang JIT)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"  # libnuma.so.1 for sgl_kernel
export TMPDIR=/workspace/shenchengyu/tmp
export HF_HOME=/workspace/shenchengyu/hf-home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MCP_CONFIG_PATH=EnvFactory/configs/mcp_server.json   # MCP tool servers for rollout
export LOGGING_LEVEL=WARN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

########################### user-adjustable ###########################
INFER_BACKEND=sglang
# SFT weights + canonical base tokenizer (so transformers 4.55.4 can load the tokenizer;
# see prepare_rl_init.py). Direct checkpoint-104 path fails: it was saved by tf 5.8.0.
MODEL_PATH=/workspace/shenchengyu/yizhigao/envfactory_repro/models/rl_init/qwen3-1.7b-sft-ep1
export CUDA_VISIBLE_DEVICES=1
NGPUS_PER_NODE=1

train_files=data/env_factory_rl.json
val_files=data/env_factory_rl.json        # reused (validation disabled below)

train_batch_size=256                      # == run_qwen3_8b.sh
ppo_mini_batch_size=64                    # == run_qwen3_8b.sh
ppo_micro_batch_size_per_gpu=2            # == run_qwen3_8b.sh
max_prompt_length=16384                   # == run_qwen3_8b.sh
max_response_length=8192                  # == run_qwen3_8b.sh

rollout_tp=1
rollout_gpu_mem_util=0.5                  # DEV: 0.8 (8-GPU) -> 0.5 (1-GPU colocation; OOM knob #1)
rollout_n=8                               # == run_qwen3_8b.sh
rollout_temperature=0.7                   # == run_qwen3_8b.sh
rollout_log_prob_micro_batch_size_per_gpu=8   # == run_qwen3_8b.sh

total_epochs=10                           # == run_qwen3_8b.sh
save_freq=20                              # == run_qwen3_8b.sh
test_freq=-1                              # DEV: 5 -> -1 (val not released)
val_before_train=False                    # DEV: True -> False (val not released)
########################### end user-adjustable ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

REWARD=(
    # custom_reward_function is a TOP-LEVEL config key in verl 0.6.1 (no `reward:` wrapper).
    # reward_scheduler / multi_turn / ref.log_prob / enable_thinking come from the
    # mcp_factory_grpo.yaml base config loaded via --config-name below (NOT from CLI).
    custom_reward_function.path=EnvFactory/reward/tool_reward_fcn.py
    custom_reward_function.name=compute_tool_reward
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
    data.dataloader_num_workers=2            # default 8 -> too many subprocesses; container
                                             # OOM-kills a dataloader worker at ~85GB. 2 is
                                             # enough (rollout generation, not data loading,
                                             # is the bottleneck).
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_fused_kernels=True       # == run_qwen3_8b.sh (flash-attn 2.7.4.post1 installed)
    actor_rollout_ref.model.use_remove_padding=True      # == run_qwen3_8b.sh
    actor_rollout_ref.model.enable_gradient_checkpointing=True   # == run_qwen3_8b.sh
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
    # --- two 1-GPU fixes found via smoke testing (deviate from defaults; see header) ---
    # sglang loads REAL checkpoint weights instead of dummy random init. With the default
    # `dummy`, the async agent loop's weight sync doesn't deliver trained weights before
    # the first rollout -> sglang emits garbage tokens -> 0 tool calls / 0 reward.
    actor_rollout_ref.rollout.load_format=auto
    # Don't free the sglang KV cache (skip the init-time sleep). On 1 GPU the init-sleep
    # races sglang HTTP readiness (release_memory_occupation connection error). 1.7B +
    # gpu_mem_util=0.5 leaves enough room for sglang and the actor to coexist.
    actor_rollout_ref.rollout.free_cache_engine=False
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

PROJECT_NAME=EnvFactory
EXPERIMENT_NAME=${PROJECT_NAME}-RL
TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=1
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.val_before_train=${val_before_train}
    trainer.total_epochs=${total_epochs}
    trainer.validation_data_dir=log_validation/${EXPERIMENT_NAME}
    trainer.rollout_data_dir=log_rollout/${EXPERIMENT_NAME}
)

# Base config = the authors' mcp_factory_grpo.yaml (provides multi_turn / ref.log_prob /
# reward_scheduler / enable_thinking / trust_remote_code). CLI below only sets run-time
# variables (paths, batch, GPU count, Option-B val, etc.).
python3 -m verl.trainer.main_ppo \
    --config-name mcp_factory_grpo \
    --config-path "$VERL_DIR/EnvFactory/configs" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
