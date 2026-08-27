#!/usr/bin/env bash
# ============================================================
# spbench_v1 trajectory rollout 采集入口(教师模型在 env 里跑,采多轮轨迹;smoke/full 同一脚本,参数区分)
#
# 原理:调 OmniaBench 官方 harness 的 SFT env(run_eval.py
#   --env-name omniabench_conversation_sft),教师模型在可执行
#   环境里多轮 tool call,harness 记录 env_trajectory 并跑
#   rubric 评分,轨迹产物落 data/trajectories/spbench_v1/。
#
# 用法:
#   export AGENT_API_KEY=sk-xxx        # yibuapi key(必填)
#   ./src/traj/rollout_spbench.sh --mode verify  # 1 条快速验证(三态 think 检查等)
#   ./src/traj/rollout_spbench.sh --mode smoke   # 3 条冒烟
#   ./src/traj/rollout_spbench.sh --mode full    # 全量 1102 条
#
# 可选覆盖:
#   MODEL=deepseek-v4-pro       agent + user 模拟器同款(负责人指定)
#   JUDGE_MODEL=xxx             rubric judge,默认与 MODEL 同款
# 注:三态思考控制(负责人要求:agent 开,user/rubric 关)
#   agent : --enable-thinking 显式开(yibuapi 默认也开,显式传防中转改默认后静默翻车)
#   user/judge : --thinking-config 显式关。harness 原本对 deepseek 不传思考参数 =
#               走中转默认 = 全开(2026-08-27 实测 user/judge 偷偷 think 已坐实),
#               已在 agent_llm_inference.py + user_llm_inference.py 打 patch:
#               deepseek 模型注入 thinking 参数(enabled/disabled)
# ============================================================
set -euo pipefail

MODE="smoke"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

AGENT_API_KEY="${AGENT_API_KEY:?ERROR: 先 export AGENT_API_KEY=sk-xxx (yibuapi key)}"
BASE_URL="https://yibuapi.com/v1"
MODEL="${MODEL:-deepseek-v4-pro}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"

LIMA=/workspace/shenchengyu/yizhigao/LIMA
EVAL=$LIMA/OmniaBench/evaluation

case "$MODE" in
  verify)
    GLOBAL_ID_RANGE="1-1"
    WORKERS=1
    OUT=$LIMA/data/trajectories/spbench_v1/smoke
    ;;
  smoke)
    GLOBAL_ID_RANGE="1-3"
    WORKERS=3
    OUT=$LIMA/data/trajectories/spbench_v1/smoke
    ;;
  full)
    GLOBAL_ID_RANGE="1-1102"
    WORKERS=16
    OUT=$LIMA/data/trajectories/spbench_v1/full
    ;;
  *) echo "MODE 必须是 verify|smoke|full"; exit 1;;
esac
mkdir -p "$OUT" "$LIMA/logs/spbench_v1"

# yibuapi 在美国,必须走本地 mihomo(同 tau2 经验),否则连接劣化
export http_proxy=http://127.0.0.1:7896 https_proxy=http://127.0.0.1:7896
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY=localhost,127.0.0.1,::1

cd "$EVAL"
source .venv/bin/activate
# 关键:python 输出接 pipe 后是 8KB 全缓冲,不设这个进度条会"假卡住"看不到
export PYTHONUNBUFFERED=1

LOG=$LIMA/logs/spbench_v1/rollout_${MODE}_$(date +%Y%m%d_%H%M%S).log
echo "=== spbench trajectory rollout mode=$MODE model=$MODEL range=$GLOBAL_ID_RANGE → $OUT ==="
# ---- env-name 选择(run_eval.py ENV_CLS_MAP,四个=两个维度的组合)----
#   conversation_*     多轮对话:UserAgent 模拟用户(persona 驱动),用户发 ###STOP### 终止
#   non_conversation_* 单轮多步:任务描述即开场,无用户模拟,agent 自报 "Task Completed/Failed" 终止
#   *_rl  = base_env 子类:每步 exec checklist 校验函数(check_func 对比 init/final state)算 RL reward
#   *_sft = 独立实现:不算 reward(官方注释:SFT 为省成本不合成校验函数),
#           轨迹质量事后靠 rubric judge 打分过滤
#   四者 MAX_STEPS 均为 200(源码 MAX_STEPS_MAP 为准,argparse help 里 rl=100/30 是过时文案)
#   spbench=多轮 DAG 任务 + 目的是采 SFT 轨迹 → conversation_sft
python scripts/run_eval.py \
  --env-name omniabench_conversation_sft \
  --task-items-path "$LIMA/data/raw/spbench_v1_0_prepared.json" \
  --global-id-range "$GLOBAL_ID_RANGE" \
  --agent-model "$MODEL" --agent-provider openai \
  --agent-api-key "$AGENT_API_KEY" --agent-base-url "$BASE_URL" \
  --user-model "$MODEL" --user-provider openai \
  --user-api-key "$AGENT_API_KEY" --user-base-url "$BASE_URL" \
  --rubric-judge-model "$JUDGE_MODEL" --rubric-judge-provider openai \
  --rubric-judge-api-key "$AGENT_API_KEY" --rubric-judge-base-url "$BASE_URL" \
  --lang-filter all --prompt-lang auto \
  --max-task-workers "$WORKERS" \
  --enable-thinking \
  --thinking-config '{"user_enable_thinking": false, "rubric_judge_enable_thinking": false}' \
  --out-dir "$OUT" \
  2>&1 | tee "$LOG"
