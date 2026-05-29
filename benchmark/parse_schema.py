"""
解析 swe-bench-lite 中 .graph.json 数据的 schema。

遍历所有图文件，按 nodeType / edgeType 统计:
  - 每个字段的数据类型（JSON type）
  - 字段的出现率（是否在所有实例中都存在）
  - 对字符串字段，统计其值特征（empty / "null" / 有内容 / 长度范围）
  - 对整数字段，统计取值范围
"""

from __future__ import annotations

import json
import os
import glob
import argparse
from collections import defaultdict


GRAPH_DIR = "/home/10358884/10358884/tools/code_graph_data/swe-bench-lite"


class FieldStats:
    """单个字段在所有样本上的统计信息。"""

    def __init__(self):
        self.count: int = 0                  # 出现次数
        self.is_int: bool = False            # 是否所有值都是 int
        self.is_str: bool = False            # 是否所有值都是 str
        self.is_bool: bool = False
        self.is_null: bool = False            # 出现过 null (Python None)
        self.has_empty_str: bool = False      # 出现过空字符串
        self.has_null_str: bool = False       # 出现过字符串 "null"
        self.has_content_str: bool = False    # 出现过有内容的字符串
        self.int_min: int = 0
        self.int_max: int = 0
        self.str_min_len: int = 0
        self.str_max_len: int = 0
        self.sample_values: list[str] = []    # 最多保留 5 个样本值

    def update(self, value, max_samples: int = 5):
        self.count += 1
        if value is None:
            self.is_null = True
        elif isinstance(value, bool):
            self.is_bool = True
        elif isinstance(value, int) and not isinstance(value, bool):
            self.is_int = True
            if self.count == 1:
                self.int_min = self.int_max = value
            else:
                self.int_min = min(self.int_min, value)
                self.int_max = max(self.int_max, value)
        elif isinstance(value, float):
            pass  # 当前数据中未出现 float
        elif isinstance(value, str):
            self.is_str = True
            if value == "":
                self.has_empty_str = True
            elif value == "null":
                self.has_null_str = True
            else:
                self.has_content_str = True
            l = len(value)
            if self.count == 1:
                self.str_min_len = self.str_max_len = l
            else:
                self.str_min_len = min(self.str_min_len, l)
                self.str_max_len = max(self.str_max_len, l)
            if len(self.sample_values) < max_samples and value not in self.sample_values:
                self.sample_values.append(value)

    def describe_type(self) -> str:
        """生成人类可读的类型描述。"""
        parts = []
        if self.is_int:
            parts.append("int")
            if self.int_min != self.int_max:
                parts[-1] += f" (range {self.int_min}..{self.int_max})"
        if self.is_str:
            modifiers = []
            if self.has_null_str:
                modifiers.append('can be "null"')
            if self.has_empty_str:
                modifiers.append("can be empty")
            if self.has_content_str:
                lo = self.str_min_len
                hi = self.str_max_len
                if lo == hi:
                    modifiers.append(f"content len={lo}")
                else:
                    modifiers.append(f"content len {lo}..{hi}")
            if modifiers:
                parts.append("str: " + ", ".join(modifiers))
            else:
                parts.append("str")
        if self.is_bool:
            parts.append("bool")
        if self.is_null:
            parts.append("null")
        if not parts:
            parts.append("unknown")
        return " | ".join(parts)


def analyze_graph_file(filepath: str, total_nodes: int) -> tuple[dict, dict]:
    """分析单个图文件，返回 node/edge schema 统计。"""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    node_schema: dict[str, dict[str, FieldStats]] = defaultdict(lambda: defaultdict(FieldStats))
    edge_schema: dict[str, dict[str, FieldStats]] = defaultdict(lambda: defaultdict(FieldStats))

    for node in data.get("nodes", []):
        nt = node.get("nodeType", "?")
        for field, value in node.items():
            node_schema[nt][field].update(value)

    for edge in data.get("edges", []):
        et = edge.get("edgeType", "?")
        for field, value in edge.items():
            edge_schema[et][field].update(value)

    return dict(node_schema), dict(edge_schema)


def merge_stats(target: dict, source: dict):
    """将 source 的 FieldStats 合并到 target。"""
    for type_name, fields in source.items():
        for field_name, stats in fields.items():
            # FieldStats 不支持简单合并，用累加方式
            existing = target[type_name].get(field_name)
            if existing is None:
                target[type_name][field_name] = stats
            else:
                # 合并基础属性
                existing.count += stats.count
                existing.is_int = existing.is_int or stats.is_int
                existing.is_str = existing.is_str or stats.is_str
                existing.is_bool = existing.is_bool or stats.is_bool
                existing.is_null = existing.is_null or stats.is_null
                existing.has_empty_str = existing.has_empty_str or stats.has_empty_str
                existing.has_null_str = existing.has_null_str or stats.has_null_str
                existing.has_content_str = existing.has_content_str or stats.has_content_str
                if stats.is_int:
                    existing.int_min = min(existing.int_min, stats.int_min)
                    existing.int_max = max(existing.int_max, stats.int_max)
                if stats.is_str:
                    existing.str_min_len = min(existing.str_min_len, stats.str_min_len)
                    existing.str_max_len = max(existing.str_max_len, stats.str_max_len)
                # 合并样本值
                for sv in stats.sample_values:
                    if sv not in existing.sample_values and len(existing.sample_values) < 5:
                        existing.sample_values.append(sv)


def main():
    parser = argparse.ArgumentParser(description="Parse schema of .graph.json files")
    parser.add_argument("--graph-dir", default=GRAPH_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Only scan first N files")
    parser.add_argument("--output", default=None, help="Save schema JSON to file")
    parser.add_argument("--sample-values", type=int, default=3,
                        help="Number of sample values to show per field")
    args = parser.parse_args()

    graph_files = sorted(glob.glob(os.path.join(args.graph_dir, "*.graph.json")))
    if not graph_files:
        print(f"No .graph.json files found in {args.graph_dir}")
        return

    if args.limit:
        graph_files = graph_files[: args.limit]

    print(f"Analyzing {len(graph_files)} graph files...")

    # 全局合并
    merged_nodes: dict[str, dict[str, FieldStats]] = defaultdict(lambda: defaultdict(FieldStats))
    merged_edges: dict[str, dict[str, FieldStats]] = defaultdict(lambda: defaultdict(FieldStats))
    total_nodes = 0
    total_edges = 0

    for i, gf in enumerate(graph_files):
        ns, es = analyze_graph_file(gf, total_nodes)
        merge_stats(merged_nodes, ns)
        merge_stats(merged_edges, es)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(graph_files)}] ...")

    print(f"\nDone. Scanned {len(graph_files)} files.")

    # --- 输出 ---
    print()
    print("=" * 70)
    print("NODE SCHEMA  (field: type  [occurrence / total of this nodeType])")
    print("=" * 70)

    node_order = ["Repo", "Package", "File", "TextFile", "Class", "Function", "Attribute", "Lambda"]
    for nt in node_order:
        if nt not in merged_nodes:
            continue
        fields = merged_nodes[nt]
        # 获取该类型节点的总出现次数 (用 id 字段的 count)
        total = fields.get("id", list(fields.values())[0]).count
        print(f"\n## {nt}  ({total} instances)")
        for field in sorted(fields.keys()):
            stats = fields[field]
            pct = stats.count / total * 100
            occ = "" if stats.count == total else f" [{stats.count}/{total} = {pct:.0f}%]"
            type_desc = stats.describe_type()
            print(f"    {field}: {type_desc}{occ}")
            if stats.sample_values:
                previews = []
                for sv in stats.sample_values[: args.sample_values]:
                    sv_short = sv.replace("\n", "\\n")
                    if len(sv_short) > 60:
                        sv_short = sv_short[:57] + "..."
                    previews.append(f'"{sv_short}"')
                print(f"        samples: {', '.join(previews)}")

    print()
    print("=" * 70)
    print("EDGE SCHEMA")
    print("=" * 70)

    edge_order = ["contains", "calls", "imports"]
    for et in edge_order:
        if et not in merged_edges:
            continue
        fields = merged_edges[et]
        total = fields.get("edgeType", list(fields.values())[0]).count
        print(f"\n## {et}  ({total} instances)")
        for field in sorted(fields.keys()):
            stats = fields[field]
            pct = stats.count / total * 100
            occ = "" if stats.count == total else f" [{stats.count}/{total} = {pct:.0f}%]"
            print(f"    {field}: {stats.describe_type()}{occ}")

    # --- JSON 输出 ---
    if args.output:
        output = {}
        for scope_name, merged in [("nodes", merged_nodes), ("edges", merged_edges)]:
            output[scope_name] = {}
            for type_name, fields in merged.items():
                total = fields.get("id", fields.get("edgeType", list(fields.values())[0])).count
                output[scope_name][type_name] = {
                    "count": total,
                    "fields": {},
                }
                for fname, stats in fields.items():
                    entry = {"type": stats.describe_type()}
                    if stats.count != total:
                        entry["occurrence"] = f"{stats.count}/{total}"
                    if stats.sample_values:
                        entry["samples"] = stats.sample_values[: args.sample_values]
                    output[scope_name][type_name]["fields"][fname] = entry
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(output, fh, ensure_ascii=False, indent=2)
        print(f"\nSchema saved → {args.output}")


if __name__ == "__main__":
    main()
