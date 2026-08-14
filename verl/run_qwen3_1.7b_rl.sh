#!/usr/bin/env bash
# GRPO RL for EnvFactory-1.7B reproduction | SGLang rollout | 2×H100 (80GB)
# Faithful port of the verl fork's run_qwen3_8b.sh (release/v0.6.1, commit 99f4e378).
# All algorithmic / reward / batch hyperparameters are kept IDENTICAL to run_qwen3_8b.sh.
# Deviations are only what 2 GPUs + unreleased val file force (see table below).
#
#   PARAM                        run_qwen3_8b.sh     HERE        reason
#   NGPUS_PER_NODE               8                   2           only 2 GPUs available
#   rollout.gpu_memory_utilization 0.8               0.7         colocation headroom (OOM knob #1)
#   rollout.load_format          dummy (default)     auto        !! see BLOCKER below --
#   rollout.free_cache_engine    True  (default)     False       !! these two are ONE bug,
#                                                                   not two deviations, and
#                                                                   they break RL learning
#     NOTE: run_qwen3_8b.sh does not mention either key (grep: 0 hits). The "official"
#     column above is the verl 0.6.1 upstream DEFAULT (rollout.py:137 free_cache_engine=True,
#     rollout.py:189 load_format="dummy"), i.e. the authors never had to touch these --
#     not that they deliberately chose True.
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
# !!! BLOCKER -- DO NOT PUBLISH NUMBERS FROM THIS SCRIPT AS-IS !!!
# free_cache_engine=False silently disables sglang weight resync in async rollout mode, so
# every rollout is sampled from the policy frozen at init while the actor trains on. See the
# long comment at the free_cache_engine line in ROLLOUT= below for the exact call chain.
# The pipeline runs end-to-end and produces plausible-looking rewards, which is precisely
# why this is dangerous. Fix free_cache_engine=True (init-time release_memory_occupation
# connection error) before treating any result as a reproduction.
#
# Smoke-test status (mechanically validated only): full pipeline runs end-to-end (rollout ->
# reward -> PPO -> save). load_format=auto makes sglang generate coherent multi-turn tool
# calls (verified in the dump). Watch at scale: (1) gpu_memory_utilization if OOM; (2) CPU
# RAM ~85GB -> the DataLoader worker gets OOM-killed at end-of-run (harmless for completed
# steps, but monitor for long runs); (3) reward signal -- epoch-1 model gives 0 reward on a
# tiny smoke sample, confirm nonzero reward appears in the first steps of the full run.
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
MODEL_PATH=/workspace/shenchengyu/yizhigao/envfactory_repro/models/qwen3-1.7b-sft-ep1
# Overridable from the environment so a smoke test can be pinned to a free GPU while a
# real run occupies another, e.g.  CUDA_VISIBLE_DEVICES=0 bash run_qwen3_1.7b_rl.sh ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"

train_files=data/env_factory_rl.json
val_files=data/env_factory_rl.json        # reused (validation disabled below)

train_batch_size=256                      # == run_qwen3_8b.sh
ppo_mini_batch_size=64                    # == run_qwen3_8b.sh
ppo_micro_batch_size_per_gpu=2            # == run_qwen3_8b.sh
max_prompt_length=16384                   # == run_qwen3_8b.sh
max_response_length=8192                  # == run_qwen3_8b.sh

rollout_tp=1
rollout_gpu_mem_util=0.8                  # DEV: 0.8 (8-GPU) -> 0.5 (1-GPU colocation; OOM knob #1)
rollout_n=8                               # == run_qwen3_8b.sh
rollout_temperature=0.7                   # == run_qwen3_8b.sh
rollout_log_prob_micro_batch_size_per_gpu=8   # == run_qwen3_8b.sh (restored from 2 after FSDP2 fix)
                                          # sglang holding 65.92/79.1GiB, so 8 of them asked
                                          # for 1.38GiB with 525MiB free. It runs under
                                          # no_grad -- a smaller micro-batch only adds loop
                                          # iterations, the log probs are unchanged.
                                          # NOTE: the REF model has its own knob of the same
                                          # name in mcp_factory_grpo.yaml; both were lowered.

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
    # --- WARNING: the two settings below are NOT two independent workarounds. ---
    # They are one bug (free_cache_engine=True dies in init) plus the thing that hides it.
    # Together they make this run SCIENTIFICALLY INVALID as an RL reproduction. Read on
    # before trusting any reward curve produced with these values.
    #
    # Why: with rollout.mode=async, ray_trainer.py:783 sets async_rollout_mode=True, so
    # rollout only ever goes through async_rollout_manager.generate_sequences
    # (ray_trainer.py:1053). The FSDP worker's own generate_sequences (fsdp_workers.py:926,
    # which calls rollout_mode() -> update_weights()) is NEVER invoked in this mode.
    # In async mode the ONLY path to rollout_mode()/update_weights() is
    # AgentLoopManager.wake_up() -- and agent_loop.py:759-760 gates that on
    # `if free_cache_engine:`. (Checked: rollout_mode() has exactly 4 call sites repo-wide;
    # fsdp_workers.py:1931, reached from wake_up, is the only async one.)
    #
    # => With free_cache_engine=False, sglang's weights are frozen at whatever init loaded
    #    and are never resynced from the training actor. The actor does keep updating
    #    (actor/grad_norm ~0.19), but every rollout is sampled from the FROZEN policy.
    #    GRPO then computes advantages over samples from a policy that never improves.
    # NOTE: this is established by reading the code paths above, NOT by measurement. The
    #    4 logged steps show critic/score/mean 0.439/0.457/0.420/0.429 -- flat, consistent
    #    with the above, but 4 steps is far too short to be evidence either way.
    #
    # free_cache_engine=True is the CORRECT value and enables weight synchronization.
    # SOLUTION: Use FSDP2 instead of FSDP (default). FSDP2 uses different communication
    # mechanisms that avoid the pidfd_getfd system call, which fails in containers without
    # CAP_SYS_PTRACE capability (RuntimeError: pidfd_getfd: Operation not permitted).
    # Tested 2026-08-13: FSDP2 + free_cache_engine=True completes training successfully.
    # Reference: https://github.com/verl-project/verl/issues/3377
    # Reference: https://github.com/verl-project/verl/issues/2846
    actor_rollout_ref.rollout.free_cache_engine=True
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# FSDP2 configuration to fix pidfd_getfd permission error in containers
# FSDP2 uses different communication mechanisms that don't require CAP_SYS_PTRACE
# This is required for free_cache_engine=True to work in restricted containers
STRATEGY=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.ref.strategy=fsdp2
    critic.strategy=fsdp2
    reward_model.strategy=fsdp2
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
    "${STRATEGY[@]}" \
    "${TRAINER[@]}" \
    "$@"
