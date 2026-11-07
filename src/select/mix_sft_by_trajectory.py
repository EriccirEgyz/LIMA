#!/usr/bin/env python3
"""Mix converted Alpaca SFT pools by trajectory quotas, using adjacent manifests.

Only the Python standard library is required. See README.md for the grouping
contract, reproducibility rules, and an example. --dry-run never writes files.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    manifest: Path
    count: int


@dataclass(frozen=True)
class Group:
    manifest_index: int
    row: dict
    start: int
    end: int

    @property
    def identity(self):
        # global_id identifies a task, not a particular sampled trajectory.
        return (self.row["source"], self.row["task_index"], self.row["sample_idx"])


@dataclass
class Pool:
    source: Source
    groups: list
    manifest_rows: int
    sample_count: int
    manifest_sha256: str


def parse_sources(values):
    sources, files = [], set()
    for filename, count_text in values:
        name = Path(filename).stem
        try:
            count = int(count_text)
        except ValueError:
            raise ValueError(f"{name}: 轨迹数量必须是正整数: {count_text!r}") from None
        if count <= 0:
            raise ValueError(f"{name}: 轨迹数量必须是正整数: {count}")
        path = Path(filename).expanduser().resolve(strict=True)
        if path.suffix != ".json" or not path.is_file():
            raise ValueError(f"{name}: 输入必须是 .json 文件: {path}")
        stat = path.stat()
        file_id = (stat.st_dev, stat.st_ino)
        if file_id in files:
            raise ValueError(f"{name}: 重复的输入文件（包括软链/硬链别名）: {path}")
        manifest = path.with_suffix(".manifest.jsonl")
        if not manifest.is_file():
            raise ValueError(f"{name}: 缺少配套 manifest，无法按轨迹抽样: {manifest}")
        sources.append(Source(path.stem, path, manifest, count))
        files.add(file_id)
    # Labels are for display/auditing only. Distinguish equal basenames without
    # requiring user-supplied names or coupling labels to the random seed.
    names = Counter(source.name for source in sources)
    return [Source(source.name if names[source.name] == 1 else str(source.path),
                   source.path, source.manifest, source.count) for source in sources]


def load_pool(source):
    raw = source.manifest.read_bytes()
    groups, identities = [], set()
    offset = row_count = 0
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{source.name}: manifest 第 {line_number} 行 JSON 无效: {exc}") from exc
        where = f"{source.name}: manifest 第 {line_number} 行"
        if not isinstance(row, dict):
            raise ValueError(f"{where} 必须是对象")
        count = row.get("n_samples")
        if type(count) is not int or count < 0 or row.get("kept") is not (count > 0):
            raise ValueError(f"{where} 的 n_samples/kept 不一致或类型无效")
        if not isinstance(row.get("source"), str) or not row["source"]:
            raise ValueError(f"{where} 缺少原始轨迹 source")
        for key in ("task_index", "sample_idx"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"{where} 的 {key} 必须是非负整数")
        if not isinstance(row.get("trajectory_mixing", []), list):
            raise ValueError(f"{where} 的 trajectory_mixing 必须是列表")
        group = Group(row_count, row, offset, offset + count)
        if group.identity in identities:
            raise ValueError(f"{where} 重复记录同一轨迹: {group.identity}")
        identities.add(group.identity)
        if count:
            groups.append(group)
        offset += count
        row_count += 1
    return Pool(source, groups, row_count, offset, hashlib.sha256(raw).hexdigest())


def select_groups(pool, seed):
    # Reset the RNG for each source so other sources cannot consume its state.
    # Labels/paths do not affect selection; fixed permutations give nested sets.
    rng = random.Random(seed)
    order = list(range(len(pool.groups)))
    rng.shuffle(order)
    return [pool.groups[i] for i in sorted(order[:pool.source.count])]


def iter_samples(path, hasher, chunk_size=1024 * 1024):
    """Read a JSON array one object at a time and hash the original UTF-8 bytes.

    Buffer size is bounded by a chunk plus the largest example for valid input.
    Repeated histories make the full DAG file too large for json.load().
    """
    def reject_constant(value):
        raise ValueError(f"非标准 JSON 常量: {value}")

    decoder = json.JSONDecoder(parse_constant=reject_constant)
    with path.open(encoding="utf-8", newline="") as stream:
        buffer, pos, eof = "", 0, False

        def more():
            nonlocal buffer, pos, eof
            chunk = stream.read(chunk_size)
            hasher.update(chunk.encode("utf-8"))
            buffer, pos = buffer[pos:] + chunk, 0
            eof = not chunk

        def peek():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n":
                    pos += 1
                if pos < len(buffer):
                    return buffer[pos]
                if eof:
                    return None
                more()

        if peek() != "[":
            raise ValueError(f"{path}: SFT 数据必须是 JSON 数组")
        pos += 1
        if peek() == "]":
            pos += 1
        else:
            while True:
                if peek() != "{":
                    raise ValueError(f"{path}: 数组中必须是样本对象，且不能有尾随逗号")
                while True:
                    try:
                        sample, end = decoder.raw_decode(buffer, pos)
                        break
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise ValueError(f"{path}: JSON 无效或不完整: {exc}") from exc
                        more()
                pos = end
                yield sample
                separator = peek()
                if separator == "]":
                    pos += 1
                    break
                if separator != ",":
                    raise ValueError(f"{path}: 样本后缺少 ',' 或 ']'")
                pos += 1
        if peek() is not None:
            raise ValueError(f"{path}: JSON 数组后存在多余内容")


def validate_sample(sample, source_name, index):
    where = f"{source_name}: SFT 样本 {index}（从 0 开始）"
    for field in ("instruction", "input", "output", "system"):
        if not isinstance(sample.get(field), str):
            raise ValueError(f"{where} 的 {field} 必须是字符串")
    history = sample.get("history")
    if not isinstance(history, list) or any(
        not isinstance(pair, list) or len(pair) != 2
        or any(not isinstance(text, str) for text in pair) for pair in history
    ):
        raise ValueError(f"{where} 的 history 必须是 [[user, assistant], ...] 字符串列表")


def prepare_mixture(sources, seed):
    pools = [load_pool(source) for source in sources]
    shortages = [
        f"{p.source.name}: 请求 {p.source.count} 条，清洗后可用 {len(p.groups)} 条，"
        f"缺 {p.source.count - len(p.groups)} 条"
        for p in pools if p.source.count > len(p.groups)
    ]
    if shortages:
        raise ValueError("轨迹数量不足（不会重复采样或写出文件）:\n" + "\n".join(shortages))
    selections = [select_groups(pool, seed) for pool in pools]
    seen = {}
    for pool, groups in zip(pools, selections):
        for group in groups:
            if group.identity in seen:
                raise ValueError(
                    f"{pool.source.name} 与 {seen[group.identity]} 重复选中同一原始轨迹: "
                    f"{group.identity}；请使用互不重叠的来源池"
                )
            seen[group.identity] = pool.source.name

    samples, manifests, summaries, input_hashes = [], [], [], {}
    for pool, groups in zip(pools, selections):
        source = pool.source
        start_output = len(samples)
        hasher = hashlib.sha256()
        before = source.path.stat()
        group_index = 0
        count = 0
        for index, sample in enumerate(iter_samples(source.path, hasher)):
            validate_sample(sample, source.name, index)
            count += 1
            while group_index < len(groups) and index >= groups[group_index].end:
                group_index += 1
            if group_index < len(groups) and index >= groups[group_index].start:
                samples.append(sample)
        after = source.path.stat()
        signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if signature(before) != signature(after):
            raise ValueError(f"{source.name}: 读取期间输入文件发生变化，请在转换完成后重试")
        if count != pool.sample_count:
            raise ValueError(
                f"{source.name}: manifest 覆盖 {pool.sample_count} 个切片，SFT 实际有 {count} 个；"
                "必须使用同一次转换/筛选生成的数据与 manifest"
            )
        digest = hasher.hexdigest()
        if digest in input_hashes:
            raise ValueError(f"{source.name} 与 {input_hashes[digest]} 的 SFT 文件内容完全相同")
        input_hashes[digest] = source.name
        selected_records = []
        offset = start_output
        for group in groups:
            end = offset + group.end - group.start
            record = {
                "source_name": source.name,
                "source_dataset": str(source.path),
                "source_manifest": str(source.manifest),
                "source_manifest_index": group.manifest_index,
                "source_index_range": [group.start, group.end],
                "output_index_range": [offset, end],
                "seed": seed,
            }
            row = dict(group.row)
            row["trajectory_mixing"] = [*row.get("trajectory_mixing", []), record]
            manifests.append(row)
            selected_records.append({
                "source": row["source"], "task_index": row["task_index"],
                "sample_idx": row["sample_idx"], "global_id": row.get("global_id"),
                "n_samples": row["n_samples"], **record,
            })
            offset = end
        summaries.append({
            "name": source.name, "dataset": str(source.path), "dataset_sha256": digest,
            "manifest": str(source.manifest), "manifest_sha256": pool.manifest_sha256,
            "requested_trajectories": source.count, "available_trajectories": len(pool.groups),
            "selected_trajectories": len(groups), "manifest_rows": pool.manifest_rows,
            "available_samples": pool.sample_count, "selected_samples": len(samples) - start_output,
            "selected": selected_records,
        })
    for summary in summaries:
        summary["trajectory_fraction"] = summary["selected_trajectories"] / len(manifests)
        summary["sample_fraction"] = summary["selected_samples"] / len(samples)
    report = {
        "format_version": 1, "seed": seed, "python_version": platform.python_version(),
        "sampling": "per-source Random(seed).shuffle; take quota prefix; labels and paths excluded",
        "ordering": "source argument order, then original manifest/sample order; no slice shuffle",
        "total_trajectories": len(manifests), "total_samples": len(samples), "sources": summaries,
    }
    return samples, manifests, report


def output_paths(filename, sources):
    output = Path(filename).expanduser().absolute()
    if output.suffix != ".json":
        raise ValueError("--out 必须以 .json 结尾")
    paths = (output, output.with_suffix(".manifest.jsonl"), output.with_suffix(".mixture.json"))
    protected = {p.resolve() for source in sources for p in (source.path, source.manifest)}
    for path in paths:
        if path.resolve() in protected:
            raise ValueError(f"输出路径不能替换输入文件: {path}")
        if os.path.lexists(path):
            raise ValueError(f"拒绝覆盖已有输出（含软链）: {path}")
    ancestor = output.parent
    while not os.path.lexists(ancestor):
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ValueError(f"输出目录的父路径不是目录: {ancestor}")
    return paths


def write_outputs(paths, samples, manifests, report):
    # All validation has finished. Exclusive opens also protect against another
    # process creating any output between preflight and writing.
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with ExitStack() as stack:
            streams = []
            for path in paths:
                streams.append(stack.enter_context(path.open("x", encoding="utf-8")))
                created.append(path)
            streams[0].write("[\n")
            for index, sample in enumerate(samples):
                if index:
                    streams[0].write(",\n")
                json.dump(sample, streams[0], ensure_ascii=False)
            streams[0].write("\n]\n")
            for row in manifests:
                streams[1].write(json.dumps(row, ensure_ascii=False) + "\n")
            json.dump(report, streams[2], ensure_ascii=False, indent=2)
            streams[2].write("\n")
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="按完整轨迹数量混合已转换/筛选的 Alpaca SFT 数据")
    parser.add_argument("--source", action="append", nargs=2, required=True,
                        metavar=("SFT_JSON", "COUNT"),
                        help="SFT JSON、所需轨迹数；可重复，自动读取同名 .manifest.jsonl，标签取自文件名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42），每个来源单独使用此 seed")
    parser.add_argument("--out", required=True, help="输出 .json，同时生成 .manifest.jsonl 和 .mixture.json")
    parser.add_argument("--dry-run", action="store_true", help="完整校验并抽样，只打印统计，不创建目录或文件")
    args = parser.parse_args(argv)
    try:
        sources = parse_sources(args.source)
        paths = output_paths(args.out, sources)
        samples, manifests, report = prepare_mixture(sources, args.seed)
        report["outputs"] = dict(zip(("dataset", "manifest", "mixture"), map(str, paths)))
        print(f"seed={args.seed}；按轨迹无放回抽样；切片保持来源内原始顺序")
        print("来源\t可用轨迹\t选中轨迹\t选中切片\t轨迹占比\t切片占比")
        for item in report["sources"]:
            print(f"{item['name']}\t{item['available_trajectories']}\t{item['selected_trajectories']}"
                  f"\t{item['selected_samples']}\t{item['trajectory_fraction']:.2%}"
                  f"\t{item['sample_fraction']:.2%}")
        print(f"合计: {len(manifests)} 条轨迹，{len(samples)} 个 SFT 切片。轨迹占比不等于训练 token 占比。")
        if args.dry_run:
            print("[dry-run] 校验通过；未创建目录或文件。")
        else:
            write_outputs(paths, samples, manifests, report)
            for path in paths:
                print(f"[写出] {path}")
        return 0
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
