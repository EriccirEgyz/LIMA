#!/usr/bin/env bash
# ============================================================
# dag_s (OmniaBench route4 / DAG-S 单轮) trajectory rollout 采集入口
#   (教师模型在 env 里跑,采单轮多步轨迹;smoke/full 同一脚本,参数区分)
#
# 原理:调 OmniaBench 官方 harness 的 run_eval.py,
#   --env-name omniabench_non_conversation_rl —— 这是 configs/routes.json
#   里 route4 的官方注册 env。单轮多步语义:任务描述即开场 user 消息,
#   无用户模拟器,agent 纯文本回复 "Task Completed" 结束(parse_action
#   Case 2 转 chat_with_user 终止)。harness 记录 messages/trajectory 并跑
#   rubric judge 评分,产物落 data/trajectories/dag_s/。
#
# 为什么走 RL env 而不是 *_sft(2026-08-31 逐行 diff 后的决策):
#   1. route4 是平铺结构(candidate_tools 直接挂 task 上,无 env_item),
#      只有 RL 系 base_env.py:124 认 candidate_tools;SFT 单轮 env 是份
#      旧拷贝只认 tools → 直接崩(所以本脚本无需任何数据预处理)
#   2. RL env 终止语义更正确:打满 MAX_STEPS 正确标 truncated/MAX_TURNS
#      (SFT env 会伪装成 COMPLETED 污染筛选);解析失败计数防空转烧钱;
#      工具参数按方法签名清洗(业务性报错照样进轨迹,探索-恢复信号不丢)
#   3. reward 对 route4 是 no-op(任务没带 checklist → calculate_reward
#      返回 0.0,无崩溃无成本),评分与 spbench 同路:rubric judge
#   4. 采集动力学 = 官方评测动力学(route4/route2 评测都走这个 env)
#   代价:干净终止官方标记是 termination_reason=USER_STOP(base_env.py:306)
#        → rollout_to_sft.py 的 --accept-termination 默认值已是
#          COMPLETED,USER_STOP(两值=干净终止;存量 spbench 0 条 USER_STOP,
#          实证改默认无影响),转换时无需额外参数
#
# 用法:
#   export AGENT_API_KEY=sk-xxx        # yibuapi key(必填)
#   ./src/traj/rollout_dag_s.sh --mode verify  # 1 条快速验证(think 三态检查等)
#   ./src/traj/rollout_dag_s.sh --mode smoke   # 3 条冒烟
#   ./src/traj/rollout_dag_s.sh --mode full    # 全量 200 条
#   ./src/traj/rollout_dag_s.sh --mode full --resume   # 断点续跑
#
# 可选覆盖:
#   MODEL=deepseek-v4-pro       教师 agent(负责人指定)
#   JUDGE_MODEL=xxx             rubric judge,默认与 MODEL 同款
#   TEMPERATURE=0.0             agent 采样温度,默认 0.0 = 官方评测协议
#                               (论文 §4.1 "Temperature is set to 0 whenever
#                               supported")。⚠️ 2026-09-01 首轮 full 用的是
#                               harness 默认 0.5,pass@1 仅 20% vs 官方 DSV4-Pro
#                               DAG-S 63.5%,温度方差是头号嫌疑 —— 后续跑全用 0
#   PASS_K=7                    每 task 采样次数(默认 1)。>1 时产物带 _passk{N}
#                               后缀,与 pass_k=1 天然隔离;resume 只在同 pass_k
#                               内捞历史分片。与 pass_k=1 产物合并时注意 sample_idx
#                               都从 1 编号,需先重编号再合并(同 rollout_spbench.sh)
#   OUT_NAME=full_pass7         覆盖保存目录最后一级名(默认 verify/smoke→smoke,
#                               full→full)
# 注1:单轮模式无用户模拟器 —— --user-model 系列参数不参与推理,但 run_eval
#   的产物文件名前缀里含 user_model 名,传 $MODEL 保持文件名诚实(缺省会写成 gpt-4.1)。
# 注2:三态思考控制在此退化为两态:agent 开(--enable-thinking)、judge 关
#   (--thinking-config);user 态不存在。deepseek 的 thinking 注入 patch 在
#   agent_llm_inference.py / user_llm_inference.py,见 memory。
# 注3:route4 全 200 条都是英文;--prompt-lang auto 按每条 task 的 lang=en
#   选英文系统提示 + 英文 judge 模板(与官方一致)。
# 注4:global_id 3000000-3000221 有 22 个空洞(如 3000001),smoke 选段已避开;
#   范围按 id 匹配选不中只是跳过,无害。
# 注5:训练用这批数据后,评测不可再用 route4/DAG-S(泄露),见
#   memory omniabench-spbench-dataset-relation 的同类约束。
# ============================================================
set -euo pipefail

MODE="smoke"
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --resume) RESUME=1; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

AGENT_API_KEY="${AGENT_API_KEY:?ERROR: 先 export AGENT_API_KEY=sk-xxx (yibuapi key)}"
BASE_URL="https://yibuapi.com/v1"
MODEL="${MODEL:-deepseek-v4-pro}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
PASS_K="${PASS_K:-1}"
OUT_NAME="${OUT_NAME:-}"

LIMA=/workspace/shenchengyu/yizhigao/LIMA
EVAL=$LIMA/OmniaBench/evaluation
TASKS=$EVAL/data/routes/route4.json   # 官方 HF 数据(scuuy666/OmniaBench),orchestrate/手动已下载

case "$MODE" in
  verify)
    GLOBAL_ID_RANGE="3000000"        # 1 条
    WORKERS=1
    OUT=$LIMA/data/trajectories/dag_s/smoke
    ;;
  smoke)
    GLOBAL_ID_RANGE="3000000-3000003" # 3000001 是空洞 → 实选 3 条
    WORKERS=3
    OUT=$LIMA/data/trajectories/dag_s/smoke
    ;;
  full)
    GLOBAL_ID_RANGE="3000000-3000221" # 22 个空洞 → 实选 200 条
    WORKERS=16
    OUT=$LIMA/data/trajectories/dag_s/full
    ;;
  *) echo "MODE 必须是 verify|smoke|full"; exit 1;;
esac
# OUT 最后一级目录名可用 OUT_NAME 覆盖(默认按 mode 取 smoke/full)
if [[ -n "$OUT_NAME" ]]; then
  OUT="$LIMA/data/trajectories/dag_s/$OUT_NAME"
fi
mkdir -p "$OUT" "$LIMA/logs/dag_s"

[[ -f "$TASKS" ]] || { echo "ERROR: 缺 $TASKS(orchestrate_eval 会自动下载,或手动 huggingface_hub.snapshot_download repo=scuuy666/OmniaBench repo_type=dataset allow_routes/*)"; exit 1; }

# yibuapi 在美国,必须走本地 mihomo(同 tau2/spbench 经验)
export http_proxy=http://127.0.0.1:7896 https_proxy=http://127.0.0.1:7896
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY=localhost,127.0.0.1,::1

cd "$EVAL"
source .venv/bin/activate
# 关键:python 输出接 pipe 后是 8KB 全缓冲,不设这个进度条会"假卡住"看不到
export PYTHONUNBUFFERED=1

LOG=$LIMA/logs/dag_s/rollout_${MODE}_$(date +%Y%m%d_%H%M%S).log
echo "=== dag_s(route4/DAG-S 单轮) trajectory rollout mode=$MODE model=$MODEL range=$GLOBAL_ID_RANGE → $OUT ==="
python scripts/run_eval.py \
  --env-name omniabench_non_conversation_rl \
  --task-items-path "$TASKS" \
  --global-id-range "$GLOBAL_ID_RANGE" \
  --agent-model "$MODEL" --agent-provider openai \
  --agent-api-key "$AGENT_API_KEY" --agent-base-url "$BASE_URL" \
  --user-model "$MODEL" --user-provider openai \
  --rubric-judge-model "$JUDGE_MODEL" --rubric-judge-provider openai \
  --rubric-judge-api-key "$AGENT_API_KEY" --rubric-judge-base-url "$BASE_URL" \
  --lang-filter all --prompt-lang auto \
  --max-task-workers "$WORKERS" \
  --pass-k "$PASS_K" \
  --enable-thinking \
  --thinking-config '{"rubric_judge_enable_thinking": false}' \
  --out-dir "$OUT" \
  $( [[ "$RESUME" == 1 ]] && echo --resume ) \
  2>&1 | tee "$LOG"

echo "=== 转换 SFT 数据集(满分筛选;终止语义默认已兼容 USER_STOP,无需额外参数) ==="
echo "  cd $LIMA && python src/traj/rollout_to_sft.py --input $OUT --out data/sft/dag_s/dag_s_sft.json --min-score 1.0 --tokenizer models/Qwen3-1.7B"
