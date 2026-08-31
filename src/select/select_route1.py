#!/usr/bin/env python3
# ============================================================
# 生成 OmniaBench 论文 DAG route 354 任务名单 (实验 spbench_route1)
#
# 论文筛选规则: "For the DAG route, we select the task with the
# longest reference tool chain generated during data synthesis for
# every level-2 domain, yielding 354 DAG tasks in total."
# (每个 level-2 domain 取 reference tool chain 最长的 task, 共 354)
#
# ⚠️ 该筛选发生在官方数据合成阶段(候选池更大), spbench_v1_0.json 里
# 既无 reference tool chain 字段、unique domain_l2 也只有 346(≠354),
# 无法从 spbench 重新推导 —— 官方 route1 名单即筛选结果, 本脚本只做
# 名单提取 + 与 spbench 的交叉校验:
#   1. 名单来源: OmniaBench/evaluation/data/task_domain_map.json 的
#      route1 条目 (354 条, 每个 domain_l2 恰好 1 个 task)
#   2. 校验: global_id 无重复; 354 条全部存在于 spbench_v1_0.json;
#      domain_l2 与 spbench 一致(spbench 缺字段的以官方为准)
#   3. 输出自包含名单: global_id / task_id / env_id / domain / lang,
#      供 rollout_to_sft.py --global-ids-file 消费, 也可独立做分析
#
# global_id join 的可靠性(2026-08-30 全量验证): rollout 记录的
# task_info.global_id == task_index+1, 且 task_id 与 spbench 同
# global_id 条目 1102/1102 精确匹配(env_id 差异均为 prepare_tasks.py
# 的 env_X__ClassName 确定性改写)。
#
# 用法(在 LIMA 根目录):
#   python3 src/select/select_route1.py \
#     --out data/sft/spbench_route1/route1_tasks.json
# ============================================================
import argparse
import json
from collections import Counter
from pathlib import Path

LIMA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAIN_MAP = LIMA_ROOT / "OmniaBench/evaluation/data/task_domain_map.json"
DEFAULT_SPBENCH = LIMA_ROOT / "data/raw/spbench_v1_0.json"


def main():
    ap = argparse.ArgumentParser(description="提取官方 route1 354 任务名单并与 spbench 交叉校验")
    ap.add_argument("--domain-map", default=str(DEFAULT_DOMAIN_MAP), help="官方 task_domain_map.json")
    ap.add_argument("--spbench", default=str(DEFAULT_SPBENCH), help="spbench_v1_0.json (1102 task 规格)")
    ap.add_argument("--out", default="data/sft/spbench_route1/route1_tasks.json", help="输出名单 json 路径")
    args = ap.parse_args()

    domain_map = json.load(open(args.domain_map, encoding="utf-8"))
    route1 = [e for e in domain_map if e.get("route") == "route1"]
    if len(route1) != 354:
        print(f"[!] route1 条目 {len(route1)} != 354 (论文口径), 请检查 domain map 版本")

    # --- 校验 1: global_id 无重复; 每个 level-2 domain 恰好 1 个 task (论文性质) ---
    # 注意唯一键是 domain_path (l1>l2): domain_l2 名字单独不唯一
    # (如 'Account Manager' 同时存在于 Finance 和 Telecom 下)
    gids = [e["global_id"] for e in route1]
    assert len(set(gids)) == len(gids), "route1 global_id 有重复"
    l2_count = Counter(e["domain_path_en"] for e in route1)
    assert max(l2_count.values()) == 1, f"domain_path 出现多 task: {[k for k, v in l2_count.items() if v > 1]}"
    print(f"[1] route1: {len(route1)} tasks / {len(l2_count)} 个 level-2 domain (每 domain 恰 1 task) ✓"
          f" [domain_l2 裸名去重仅 {len({e['domain_l2_en'] for e in route1})} 个, 唯一键是 l1>l2 路径]")

    # --- 校验 2: 名单全部存在于 spbench; domain 一致 ---
    sp = {t["global_id"]: t for t in json.load(open(args.spbench, encoding="utf-8"))}
    missing = [g for g in gids if g not in sp]
    assert not missing, f"route1 有 {len(missing)} 个 global_id 不在 spbench: {missing[:10]}"
    filled = mismatch = 0
    for e in route1:
        t = sp[e["global_id"]]
        if t.get("domain_l2") is None:
            filled += 1  # spbench 缺 domain 字段, 以官方为准
        elif t["domain_l2"] != e["domain_l2_zh"]:
            mismatch += 1
            print(f"[!] domain_l2 不一致 gid={e['global_id']}: spbench={t['domain_l2']!r} vs 官方={e['domain_l2_zh']!r}")
    assert mismatch == 0, "domain_l2 与官方不一致, 需人工核对"
    print(f"[2] {len(route1)}/{len(route1)} 均在 spbench_v1_0.json 内; domain 全一致"
          f" (其中 {filled} 条 spbench 缺 domain 字段, 已用官方 map 补齐)")

    # --- 校验 3: spbench 全集的 domain 集合 == route1 的 domain 集合 (都是 354) ---
    # 用 domain_l1+'->'+domain_l2 拼(全中文, 与官方 domain_path_zh 同构)。
    # 不用 t['domain']: 该字段 81 条是英文样式 'A -> B', 与 zh 路径不可直接比
    sp_all_paths = {f"{t['domain_l1']}->{t['domain_l2']}" for t in sp.values() if t.get("domain_l2")}
    r1_zh_paths = {e["domain_path_zh"] for e in route1}
    if sp_all_paths == r1_zh_paths:
        print(f"[3] spbench 全集覆盖 {len(sp_all_paths)} 个 domain, 与 route1 的 domain 集合完全相同 ✓ "
              f"(平均每 domain {len(sp) / len(r1_zh_paths):.1f} 条 task, route1 每 domain 选 1 条)")
    else:
        only_sp, only_r1 = sp_all_paths - r1_zh_paths, r1_zh_paths - sp_all_paths
        print(f"[!] domain 集合不一致: 仅 spbench {len(only_sp)} 个 {sorted(only_sp)[:5]}, "
              f"仅 route1 {len(only_r1)} 个 {sorted(only_r1)[:5]}")

    # --- 输出自包含名单 (domain 优先官方, task_id/env_id/lang/batch 取 spbench) ---
    tasks = []
    for e in sorted(route1, key=lambda x: x["global_id"]):
        t = sp[e["global_id"]]
        tasks.append({
            "global_id": e["global_id"],
            "task_id": t["task_id"],
            # 用 prepare_tasks.py 唯一化后的形式, 与 rollout 记录/manifest 直接可 join
            "env_id": f"{t['env_id']}__{t['env_class_name']}",
            "domain_l1": e.get("domain_l1_zh") or t.get("domain_l1"),
            "domain_l2": e.get("domain_l2_zh") or t.get("domain_l2"),
            "domain_path_en": e.get("domain_path_en"),
            "split": e.get("split"),
            "lang": t.get("lang"),
            "batch": t.get("batch"),
        })
    out = {
        "experiment": "spbench_route1",
        "source": {
            "domain_map": str(Path(args.domain_map).resolve().relative_to(LIMA_ROOT)),
            "spbench": str(Path(args.spbench).resolve().relative_to(LIMA_ROOT)),
            "rule": "OmniaBench 论文 DAG route: 每 level-2 domain 取 reference tool chain 最长的 task (官方合成期筛选, 直接采用其结果)",
        },
        "n_tasks": len(tasks),
        "n_not_selected": len(sp) - len(tasks),
        "tasks": tasks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    langs = Counter(t["lang"] for t in tasks)
    print(f"[4] 名单分布: lang={dict(langs)}; 未入选 spbench task {out['n_not_selected']} 条")
    print(f"[✓] 写出 {len(tasks)} tasks → {out_path}")


if __name__ == "__main__":
    main()
