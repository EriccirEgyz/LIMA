#!/usr/bin/env python3
# ============================================================
# spbench_v1 rollout 产物 → LLaMA-Factory alpaca 格式 SFT 数据集
#
# 做四件事(与 EnvFactory 基线 mcp_factory_sft_nips.json 逐项对齐):
#   1. 合并读取: 自动发现同前缀的所有 *_incremental_shards 历史分片目录
#      (含 --resume 续跑产生的多个时间戳目录), 按 时间戳旧→新 合并,
#      同 (task_index, sample_idx) 后写覆盖 —— 复刻 run_eval.py 官方
#      resume 的 result_by_run_key 语义。也支持直接传 .runs.jsonl / 最终 .json。
#      注意: 官方结束时 _merge_incremental_files 只合并本次 run 的分片目录,
#      产出的 .runs.jsonl 会缺 resume 之前的记录, 别用它当全量。
#      可选 --global-ids-file 只转换名单内的 task(如 route1 354 名单,
#      由 src/select/select_route1.py 生成), join 键 = task_info.global_id。
#   2. 筛选: result_status=completed + termination_reason=COMPLETED +
#      rubric_eval.avg_result >= min-score + judge parse_success。
#   3. 展开为 alpaca 逐轮切片: 一条轨迹 → 每个 assistant 轮一个样本
#      (history 递增前缀, 该轮为 output)。基线 26463 条即此结构。
#   4. 格式: system 内嵌 <tools>(逐字节复刻 LLaMA-Factory QwenToolUtils 的
#      QWEN_TOOL_PROMPT); 工具调用 → <tool_call> 文本; 工具返回 → 下一轮
#      user 消息包 <tool_response>; output 带 <think>(教师 reasoning_content)。
#   non-conversational 无调用轮仅允许末轮精确 Task Completed 且 finish_reason=stop。
#   其他无调用轮 HIDE: 不作监督目标, 仍保留历史和错误反馈; 训练须 mask_history=true。
#   auto 按环境元数据/已知 system 识别模式; 未知沿用原逻辑并提示, 可显式覆盖。
#
# 用法(在 LIMA 根目录):
#   python3 src/traj/rollout_to_sft.py \
#     --input data/trajectories/spbench_v1/full \
#     --out data/sft/spbench_v1/spbench_sft.json \
#     --min-score 1.0 \
#     --tokenizer models/Qwen3-1.7B --max-tokens 16384
#   终止语义已默认兼容两类 env:COMPLETED(conv_sft) + USER_STOP(非轮 RL env)
#   = 干净终止;dag_s 产物无需额外参数。
#   加 --dry-run 只看统计不落盘。
# ============================================================
import argparse
import ast
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# 逐字节复刻 LLaMA-Factory src/llamafactory/data/tool_utils.py 的 QWEN_TOOL_PROMPT
QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)

STOP_MARKER = "###STOP###"


# ---------------- 1. 读取与合并 ----------------

def _load_shard_records(shard_path, stats):
    """读一个分片文件, 返回 [(task_index, sample_idx, result)]。跳过撕裂行。"""
    records = []
    try:
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_lines"] += 1  # 正在被并发追加的最后一行
                    continue
                ti, si, result = item.get("task_index"), item.get("sample_idx"), item.get("result")
                if ti is not None and si is not None and isinstance(result, dict):
                    records.append((int(ti), int(si), result))
    except OSError as e:
        print(f"[!] 无法读取分片 {shard_path}: {e}", file=sys.stderr)
    return records


def _iter_result_items_from_payload(payload):
    """兼容最终聚合 .json 的形态: 单 result / result 列表 / pass_k>1 的 {samples:[...]} 分组。"""
    if not payload:
        return
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("samples"), list):
            group_ti = item.get("task_index")
            for sample in item["samples"]:
                if isinstance(sample, dict):
                    if sample.get("task_index") is None and group_ti is not None:
                        sample = dict(sample, task_index=group_ti)
                    yield sample
        else:
            yield item


def collect_records(input_path, stats):
    """按输入类型合并出 {(task_index, sample_idx): result}, 同 key 后写覆盖。"""
    input_path = Path(input_path).expanduser().resolve()
    merged = {}
    dirs = []

    if input_path.is_dir():
        if input_path.name.endswith("_incremental_shards"):
            # 指到了某一个分片目录: 只用它(明确指定则尊重之)
            dirs = [input_path]
        else:
            # 父目录: 发现该前缀下所有历史分片目录(旧→新, 含 resume 产生的多个时间戳)
            dirs = sorted(p for p in input_path.glob("*_incremental_shards") if p.is_dir())
            if not dirs:
                sys.exit(f"[X] {input_path} 下没找到 *_incremental_shards 目录")
    elif str(input_path).endswith(".jsonl"):
        for ti, si, result in _load_shard_records(str(input_path), stats):
            merged[(ti, si)] = result
        stats["sources"] = [str(input_path)]
        stats["dedup_drops"] = 0
        return merged
    elif str(input_path).endswith(".json"):
        payload = json.load(open(input_path, encoding="utf-8"))
        for item in _iter_result_items_from_payload(payload):
            ti, si = item.get("task_index"), item.get("sample_idx")
            if ti is not None and si is not None:
                merged[(int(ti), int(si))] = item
        stats["sources"] = [str(input_path)]
        stats["dedup_drops"] = 0
        return merged
    else:
        sys.exit(f"[X] 不支持的输入: {input_path}(须为目录 / .jsonl / .json)")

    total_lines = 0
    for d in dirs:
        shard_files = sorted(glob.glob(str(d / "incremental_shard_*.jsonl")))
        for sp in shard_files:
            recs = _load_shard_records(sp, stats)
            total_lines += len(recs)
            for ti, si, result in recs:
                merged[(ti, si)] = result  # 目录已按时间戳升序 → 后写覆盖 = 新时间戳优先
    stats["sources"] = [str(d) for d in dirs]
    stats["dedup_drops"] = total_lines - len(merged)
    return merged


def _load_global_ids(path):
    """读任务名单 → set[global_id]。兼容: [1,2] / [{"global_id":1},...] / {"tasks":[...]} / {"global_ids":[...]}"""
    payload = json.load(open(path, encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("global_ids", "tasks", "task_ids"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    gids = set()
    for item in payload if isinstance(payload, list) else []:
        if isinstance(item, dict) and item.get("global_id") is not None:
            gids.add(int(item["global_id"]))
        elif isinstance(item, int):
            gids.add(item)
    if not gids:
        sys.exit(f"[X] 名单 {path} 里没解析到任何 global_id")
    return gids


# ---------------- 2. 单条轨迹 → alpaca 样本 ----------------

def _normalize_tool_content(content):
    """工具返回内容规整为 JSON 文本。rollout 里是 python-repr 单引号 dict, 转成标准 JSON。"""
    if not isinstance(content, str):
        content = str(content)
    s = content.strip()
    try:
        return json.dumps(ast.literal_eval(s), ensure_ascii=False)
    except Exception:
        return s


def _format_tool_calls(tool_calls):
    """assistant 工具调用 → <tool_call> 文本块(复刻 QwenToolUtils.function_formatter)。"""
    blocks = []
    for tc in tool_calls or []:
        fn = tc.get("function", {})
        name, args = fn.get("name", ""), fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = args  # 保底: 保留原始字符串
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        blocks.append(f'<tool_call>\n{{"name": {json.dumps(name, ensure_ascii=False)}, "arguments": {args}}}\n</tool_call>')
    return "\n".join(blocks)


def build_system_prompt(result):
    """system = 轨迹 system 提示 + QWEN_TOOL_PROMPT 内嵌全部工具定义(与 EnvFactory 基线同构)。"""
    msgs = result.get("messages") or []
    base = ""
    for m in msgs:
        if m.get("role") == "system":
            base = str(m.get("content") or "").strip()
            break
    tool_text = ""
    for tool in result.get("tools") or []:
        wrapped = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
        tool_text += "\n" + json.dumps(wrapped, ensure_ascii=False)
    return base + QWEN_TOOL_PROMPT.format(tool_text=tool_text), bool(base), bool(tool_text)


def resolve_interaction_mode(result, requested="auto"):
    """只用显式模式/已知环境或 system, 不按目录名、USER_STOP 或用户消息猜测。"""
    if requested != "auto":
        if requested not in {"conversational", "non-conversational"}:
            raise ValueError(f"Unknown interaction mode: {requested}")
        return requested, "override"
    env_modes = {
        "omniabench_non_conversation_rl": "non-conversational",
        "omniabench_non_conversation_sft": "non-conversational",
        "omniabench_conversation_rl": "conversational",
        "omniabench_conversation_sft": "conversational",
    }
    if result.get("env_name") in env_modes:
        return env_modes[result["env_name"]], "env_name"
    base = next((str(m.get("content") or "") for m in result.get("messages") or []
                 if m.get("role") == "system"), "")
    if (("在任务完成之前，每一步都从工具列表中选择一个工具" in base
         and "只回复 `Task Completed`" in base)
            or ("Before the task is completed, at each step, choose one tool from the tool list" in base
                and "reply only with `Task Completed`" in base)):
        return "non-conversational", "system_prompt"
    if ("主动与用户沟通，直到用户结束对话" in base
            or "proactively communicate with the user when necessary until the user ends the conversation" in base):
        return "conversational", "system_prompt"
    return "unknown", "unrecognized"


def _uses_prompt_tool_calls(result):
    """FC 无调用轮也可能省略 tool_calls 键, 必须在整条轨迹上判断调用协议。"""
    if result.get("infer_mode") in {"fc", "prompt"}:
        return result["infer_mode"] == "prompt"
    msgs = result.get("messages") or []
    if any(m.get("role") == "tool" or m.get("tool_calls") for m in msgs):
        return False
    base = next((str(m.get("content") or "") for m in msgs if m.get("role") == "system"), "")
    return "<tools>" in base and "</tools>" in base


def _prompt_action_text(message):
    """与 prompt 执行路径一致, 只判思考块之后的动作正文。"""
    text = str(message.get("content") or "")
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return "" if "<think>" in text else text.strip()


def _has_tool_call(message, prompt_mode=False):
    if message.get("tool_calls"):
        return True
    if not prompt_mode:
        return False  # FC 正文中的 <tool_call> 不会实际执行, 不能据此绕过 HIDE。
    match = re.search(r"<tool_call>(.*?)</tool_call>", _prompt_action_text(message), re.DOTALL)
    if not match:
        return False
    try:
        call = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False
    return isinstance(call, dict) and "name" in call and "arguments" in call


def _finish_reason(message):
    """读取该次生成的结束原因; 不用轨迹 truncated/USER_STOP 代替。"""
    raw = message.get("raw_response")
    if isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            reason = choices[0].get("finish_reason")
            return reason if isinstance(reason, str) and reason else None
    reason = message.get("finish_reason")
    return reason if isinstance(reason, str) and reason else None


def _no_call_hide_reason(message, is_last_assistant, prompt_mode=False):
    if not is_last_assistant:
        return "intermediate_no_tool_call"
    finish_reason = _finish_reason(message)
    if finish_reason == "length":
        return "no_tool_call_length"
    if finish_reason is None:
        return "no_tool_call_unknown_finish_reason"
    if finish_reason != "stop":
        return "no_tool_call_nonstop_finish_reason"
    content = _prompt_action_text(message) if prompt_mode else str(message.get("content") or "").strip()
    if content != "Task Completed":
        return "noncanonical_completion"
    return None


def flatten_turns(result, stats, preserve_empty_assistant=False, assistant_sources=None):
    """messages → 严格交替的 [(user_text, assistant_text, reasoning)] 列表。

    - 连续 tool 消息(并行调用结果)合并为一个 user 轮, 多个 <tool_response> 块以 \\n 相接
    - ###STOP### 为 harness 控制消息, 剔除
    - reasoning 取自该 assistant 消息的 reasoning_content, 与轮次天然对齐
    - non-conversational 保留空 assistant, 作为 HIDE 历史; 不虚构正文/补思考
    - assistant_sources 可接收与返回 pairs 对齐的 (原始消息索引, 原始消息)
    - 结尾多余的非 assistant 轮裁掉; 异常交替(如连续 assistant)则整条轨迹弃用
    """
    msgs = result.get("messages") or []
    turns = []  # [(side, text, reasoning)]
    pending_tool_responses = []

    def flush_tools():
        if pending_tool_responses:
            turns.append(("user", "\n".join(pending_tool_responses), ""))
            pending_tool_responses.clear()

    for message_index, m in enumerate(msgs):
        role = m.get("role")
        if role == "system":
            continue
        elif role == "user":
            flush_tools()
            text = str(m.get("content") or "").strip()
            if text == STOP_MARKER:
                continue
            if not text:
                stats["empty_user_turns"] += 1
                text = "(empty)"
            turns.append(("user", text, ""))
        elif role == "tool":
            pending_tool_responses.append(
                f"<tool_response>\n{_normalize_tool_content(m.get('content'))}\n</tool_response>"
            )
        elif role == "assistant":
            flush_tools()
            parts = []
            text = str(m.get("content") or "").strip()
            if text:
                parts.append(text)
            tc_text = _format_tool_calls(m.get("tool_calls"))
            if tc_text:
                parts.append(tc_text)
            if not parts:
                stats["empty_assistant_msgs"] += 1
                if not preserve_empty_assistant:
                    continue
            reasoning = str(m.get("reasoning_content") or "").strip()
            turns.append(("assistant", "\n".join(parts), reasoning))
            if assistant_sources is not None:
                assistant_sources.append((message_index, m))
    flush_tools()

    # 裁掉结尾多余 user 轮(如 STOP 前后残留); 校验严格交替
    while turns and turns[-1][0] != "assistant":
        turns.pop()
        stats["tail_user_trimmed"] += 1
    if not turns or turns[0][0] != "user":
        stats["bad_alternation"] += 1
        return None
    for i, (side, _, _) in enumerate(turns):
        expect = "user" if i % 2 == 0 else "assistant"
        if side != expect:
            stats["bad_alternation"] += 1
            return None
    return [(turns[i][1], turns[i + 1][1], turns[i + 1][2]) for i in range(0, len(turns) - 1, 2)]


def trajectory_to_samples(result, stats, with_think=True, interaction_mode="auto", audit=None):
    """逐轮切片; non-conversational 的 HIDE 轮只入 history, 不生成目标。"""
    if audit is None:
        audit = {}
    mode, mode_source = resolve_interaction_mode(result, interaction_mode)
    stats[f"interaction_mode_{mode}"] += 1
    audit.update(interaction_mode=mode, mode_source=mode_source, hidden_turns=[], n_turns=0)
    non_conversational = mode == "non-conversational"
    system, has_system, has_tools = build_system_prompt(result)
    if not has_system:
        stats["no_system_prompt"] += 1
        audit["conversion_error"] = "no_system_prompt"
        return []
    if not has_tools:
        stats["no_tools"] += 1
        audit["conversion_error"] = "no_tools"
        return []
    assistant_sources = []
    pairs = flatten_turns(result, stats, preserve_empty_assistant=non_conversational,
                          assistant_sources=assistant_sources)
    if not pairs:
        audit["conversion_error"] = "flatten_failed"
        return []
    audit["n_turns"] = len(pairs)
    prompt_mode = _uses_prompt_tool_calls(result) if non_conversational else False
    if non_conversational:
        audit["call_format"] = "prompt" if prompt_mode else "fc"

    samples = []
    for i, (user_text, asst_text, think) in enumerate(pairs):
        message_index, message = assistant_sources[i]
        if non_conversational and not _has_tool_call(message, prompt_mode):
            is_last = i == len(pairs) - 1
            hide_reason = _no_call_hide_reason(message, is_last, prompt_mode)
            if hide_reason:
                finish_reason = _finish_reason(message)
                audit["hidden_turns"].append({
                    "turn": i, "message_index": message_index, "action": "HIDE",
                    "reason": hide_reason, "finish_reason": finish_reason,
                    "is_last_assistant": is_last, "empty_content": not bool(asst_text),
                    "has_reasoning": bool(think),
                })
                stats["hidden_turns"] += 1
                stats[f"hide_reason:{hide_reason}"] += 1
                if finish_reason == "length":
                    stats["hidden_length_turns"] += 1
                # pairs 保持完整, 后续 history 仍包含本轮及真实的环境反馈。
                continue
        if think:
            stats["turns_with_think"] += 1
            output = f"<think>{think}</think>\n\n{asst_text}"
        else:
            stats["turns_without_think"] += 1
            # 补空 <think></think> 保持格式一致，消除 think 泄漏风险
            if with_think:
                output = f"<think></think>\n\n{asst_text}"
            else:
                output = asst_text
        samples.append({
            "instruction": user_text,
            "input": "",
            "output": output,
            "system": system,
            "history": [[u, a] for u, a, _ in pairs[:i]],
        })
    if not samples:
        stats["no_supervised_turns"] += 1
        audit["conversion_error"] = "no_supervised_turns"
    return samples


# ---------------- 3. 筛选 ----------------

def filter_result(result, min_score, accept_termination=frozenset({"COMPLETED", "USER_STOP"})):
    """返回 (keep, reason)。

    accept_termination: 允许的 termination_reason 集合 = "干净终止"。
    conv_sft env 干净终止落 COMPLETED(spbench 语义);非轮 RL env(base_env.py)
    官方标记是 USER_STOP —— 两者等价,默认都收。其余值(INFRA_ERROR/TIMEOUT/
    MAX_TURNS/TOO_MUCH_TOOL_ERRORS/MODEL_OUTPUT_ERROR)一律拒。
    已实证对存量无影响:spbench_v1 全量 1102 条里 USER_STOP 出现 0 次。
    """
    if result.get("result_status") != "completed":
        return False, f"status={result.get('result_status')}"
    if result.get("termination_reason") not in accept_termination:
        return False, f"termination={result.get('termination_reason')}"
    if result.get("truncated"):
        return False, "truncated"
    re_ = result.get("rubric_eval") or {}
    if not re_:
        return False, "no_rubric_eval"
    meta = re_.get("judge_meta") or {}
    if meta.get("parse_success") is False:
        return False, "judge_parse_failed"
    avg = re_.get("avg_result")
    if avg is None:
        return False, "no_avg_result"
    if float(avg) < min_score - 1e-9:
        return False, f"score={avg:.3f}<{min_score}"
    if not (result.get("messages") or []):
        return False, "no_messages"
    return True, "ok"


# ---------------- 4. 统计与主流程 ----------------

def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1) + 0.5)))
    return sorted_vals[idx]


def main():
    ap = argparse.ArgumentParser(description="spbench rollout 产物 → EnvFactory 同构 alpaca SFT 数据集")
    ap.add_argument("--input", required=True,
                    help="rollout 输出父目录(自动合并所有 *_incremental_shards) / 单个分片目录 / .runs.jsonl / 最终 .json")
    ap.add_argument("--out", default="data/sft/spbench_v1/spbench_sft.json", help="输出 alpaca json 路径")
    ap.add_argument("--min-score", type=float, default=1.0, help="rubric avg_result 阈值(默认 1.0=满分)")
    ap.add_argument("--accept-termination", default="COMPLETED,USER_STOP",
                    help="允许的 termination_reason 集合,逗号分隔。默认两值并集=干净终止"
                         "(conv_sft 落 COMPLETED,非轮 RL env 落 USER_STOP,已实证存量 spbench 0 条 USER_STOP);"
                         "INFRA_ERROR/TIMEOUT/MAX_TURNS 等一律拒")
    ap.add_argument("--global-ids-file", default=None,
                    help="任务名单 json, 只保留名单内的 task(join 键 = task_info.global_id); 缺省不过滤")
    ap.add_argument("--interaction-mode", choices=("auto", "conversational", "non-conversational"),
                    default="auto", help="auto 按已知环境/system 识别; non-conversational 无调用违规轮只入 history,"
                    "不作监督目标(训练须 mask_history=true)。未知模式沿用原逻辑并提示, 可显式指定")
    ap.add_argument("--no-think", action="store_true", help="output 不拼教师 <think>(默认拼, 对齐基线)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer 路径, 用于精确 token 统计(缺省则跳过)")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="与训练 yaml cutoff_len 一致; 超长样本 LLaMA-Factory 会整条丢弃, 此处只统计预警")
    ap.add_argument("--manifest", default=None, help="逐轨迹审计清单输出路径(jsonl), 默认 out 同目录 .manifest.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    args = ap.parse_args()

    t0 = time.time()
    stats = Counter()
    accept_termination = frozenset(
        s.strip().upper() for s in args.accept_termination.split(",") if s.strip()
    ) or frozenset({"COMPLETED"})
    records = collect_records(args.input, stats)
    print(f"[1] 合并读取: {len(records)} 条唯一轨迹 (来源 {len(stats['sources'])} 个, 去重丢弃 {stats['dedup_drops']} 行, 坏行 {stats['bad_lines']})")
    for s in stats["sources"]:
        print(f"    - {s}")

    if args.global_ids_file:
        gids = _load_global_ids(args.global_ids_file)
        hit = {int((r.get("task_info") or {}).get("global_id")) for r in records.values()
               if (r.get("task_info") or {}).get("global_id") is not None}
        records = {(ti, si): r for (ti, si), r in records.items()
                   if (r.get("task_info") or {}).get("global_id") is not None
                   and int(r["task_info"]["global_id"]) in gids}
        print(f"[1b] task 过滤 {args.global_ids_file}: 名单 {len(gids)}, 命中有记录 {len(hit)}, "
              f"名单内无记录 {len(gids - hit)}, 保留记录 {len(records)}")

    manifest = []
    samples = []
    kept_trajs = 0
    reject_reasons = Counter()
    env_kept = Counter()
    env_total = Counter()

    for (ti, si), result in sorted(records.items()):
        info = result.get("task_info") or {}
        env_id = info.get("env_id", "?")
        global_id = info.get("global_id", ti)
        avg = (result.get("rubric_eval") or {}).get("avg_result")
        env_total[env_id] += 1
        keep, reason = filter_result(result, args.min_score, accept_termination)
        n_samples = 0
        audit = {}
        if keep:
            traj_samples = trajectory_to_samples(result, stats, with_think=not args.no_think,
                                                 interaction_mode=args.interaction_mode, audit=audit)
            if traj_samples:
                kept_trajs += 1
                env_kept[env_id] += 1
                n_samples = len(traj_samples)
                for s in traj_samples:
                    s["_meta"] = {"global_id": global_id, "task_id": info.get("task_id", "?"),
                                  "env_id": env_id, "turn": len(s["history"])}
                samples.extend(traj_samples)
            else:
                keep, reason = False, audit.get("conversion_error", "flatten_failed")
        if not keep:
            reject_reasons[reason.split("=")[0].split("<")[0]] += 1
        manifest.append({"source": str(Path(args.input).expanduser().resolve()),
                         "task_index": ti, "sample_idx": si,
                         "global_id": global_id, "task_id": info.get("task_id", "?"), "env_id": env_id,
                         "lang": info.get("lang"), "avg_result": avg, "kept": keep,
                         "reason": reason, "n_samples": n_samples,
                         "n_hidden_turns": len(audit.get("hidden_turns", [])), **audit})

    print(f"[2] 筛选(min_score={args.min_score}, accept_termination={sorted(accept_termination)}): "
          f"保留 {kept_trajs}/{len(records)} 条轨迹")
    for r, c in reject_reasons.most_common():
        print(f"    剔除 {r}*: {c}")
    print(f"[3] 逐轮切片: {len(samples)} 个样本 (均值 {len(samples) / max(1, kept_trajs):.1f} 样本/轨迹)")
    print(f"    think: 有 {stats['turns_with_think']} / 无 {stats['turns_without_think']}; "
          f"异常: 交替破坏 {stats['bad_alternation']}, 空 assistant {stats['empty_assistant_msgs']}, "
          f"空 user {stats['empty_user_turns']}, 尾部 user 裁剪 {stats['tail_user_trimmed']}, "
          f"无 system {stats['no_system_prompt']}, 无 tools {stats['no_tools']}")
    print(f"    模式(通过轨迹筛选后): non-conversational={stats['interaction_mode_non-conversational']}, "
          f"conversational={stats['interaction_mode_conversational']}, unknown={stats['interaction_mode_unknown']}")
    print(f"    HIDE: {stats['hidden_turns']} 轮不作目标、保留历史 (其中 length={stats['hidden_length_turns']})")
    for key, count in sorted(stats.items()):
        if key.startswith("hide_reason:"):
            print(f"      {key.split(':', 1)[1]}: {count}")
    if stats["interaction_mode_unknown"]:
        print("[!] 未识别模式的轨迹沿用原转换逻辑; 请核实或使用 --interaction-mode 显式指定。")
    if stats["hidden_turns"]:
        print("[!] HIDE 历史仍在输入中, 训练必须使用 mask_history=true, 只监督当前目标。")

    # token 统计
    if args.tokenizer and samples:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            lengths = []
            over = 0
            for s in samples:  # 粗算: 全文 token 数(无模板特殊 token, 误差 ~1-2%)
                text = s["system"] + "".join(u + a for u, a in s["history"]) + s["instruction"] + s["output"]
                n = len(tok(text, add_special_tokens=False)["input_ids"])
                lengths.append(n)
                if n > args.max_tokens:
                    over += 1
            lengths.sort()
            print(f"[4] token 统计({os.path.basename(args.tokenizer)}): p50={percentile(lengths, 0.5)} "
                  f"p90={percentile(lengths, 0.9)} p99={percentile(lengths, 0.99)} max={lengths[-1]}")
            print(f"    超过 cutoff_len={args.max_tokens} 的样本: {over} ({100 * over / len(lengths):.1f}%) "
                  f"—— 非 packing 下 LLaMA-Factory 不丢样本: mask_history 倒序装填, "
                  f"最后一轮(loss区)完整保留, 截掉最老历史(supervised.py encoded_pairs[::-1])")
        except ImportError:
            print("[4] 未装 transformers, 跳过 token 统计")
    else:
        char_lens = sorted(len(s["system"]) + sum(len(u) + len(a) for u, a in s["history"])
                           + len(s["instruction"]) + len(s["output"]) for s in samples)
        if char_lens:
            print(f"[4] 字符统计(粗估 token≈chars/3.5): p50={percentile(char_lens, 0.5) // 3.5:.0f} "
                  f"p90={percentile(char_lens, 0.9) // 3.5:.0f} max={char_lens[-1] // 3.5:.0f}")

    print(f"[5] 各环境保留率(top 8 / 共 {len(env_total)} 个 env):")
    for env, total in env_total.most_common(8):
        print(f"    {env[:60]:<60} {env_kept[env]}/{total}")

    if args.dry_run:
        print(f"[dry-run] 不落盘。总耗时 {time.time() - t0:.1f}s")
        return

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in s.items() if k != "_meta"} for s in samples]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False)
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[✓] 写出 {len(clean)} 样本 → {out}")
    print(f"[✓] 审计清单 → {manifest_path}")
    print(f"    注册: 在 LLaMA-Factory/data/dataset_info.json 加 \"spbench_sft\" 条目并 symlink 该文件(仿 env_factory_sft)")
    print(f"    总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
