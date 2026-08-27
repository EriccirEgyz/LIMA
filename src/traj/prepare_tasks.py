#!/usr/bin/env python3
"""
spbench task 预处理:env_id 唯一化。

问题:spbench_v1_0.json 内部版里 env_id 是"槽位号"而非环境身份 —— 同一 env_id
下挂着多个不同 env_class_name 的环境(1102 条 / 127 个 env_id / 557 个类名)。
OmniaBench harness 的 extract_env_items_from_task_items 按 env_id 聚合 env 代码,
后写覆盖先写 → task 要求的类在拿到的代码里不存在 →
"Class 'xxx' not found in provided env_class_code" INFRA_ERROR。

修复:把每条 task 的 env_id 改写为 f"{env_id}__{env_class_name}"。
已验证 575 个 (env_id, class_name) 组合内 env_class_code 完全一致,
同组 task 共享同一唯一键,聚合覆盖的是相同代码,无副作用。

用法:
    python src/traj/prepare_tasks.py            # 默认路径
    python src/traj/prepare_tasks.py IN OUT
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

LIMA = Path("/workspace/shenchengyu/yizhigao/LIMA")
DEFAULT_IN = LIMA / "data/raw/spbench_v1_0.json"
DEFAULT_OUT = LIMA / "data/raw/spbench_v1_0_prepared.json"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    tasks = json.loads(src.read_text(encoding="utf-8"))
    n_env_before = len({t["env_id"] for t in tasks})

    new_keys = set()
    for t in tasks:
        uniq = f"{t['env_id']}__{t['env_class_name']}"
        t["env_id"] = uniq
        new_keys.add(uniq)

    # 校验:唯一化后 key 数应等于 (env_id, class_name) 组合数,且代码组内一致
    code_by_key = defaultdict(set)
    for t in tasks:
        code_by_key[t["env_id"]].add(hash(t["env_item"]["env_class_code"]))
    assert all(len(v) == 1 for v in code_by_key.values()), "组内代码不一致,不可共享 key"

    dst.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    print(f"tasks: {len(tasks)}")
    print(f"env_id: {n_env_before} (槽位) -> {len(new_keys)} (唯一环境)")
    print(f"written: {dst} ({dst.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
