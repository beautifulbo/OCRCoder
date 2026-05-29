"""
CGM Benchmark Data Preparation

基于 swe-bench-lite 代码图谱数据集，生成评测数据：
1. 从每个图中随机选取 Function / Class 节点作为"中心节点"(seed)
2. 以 seed 为中心做 k-hop BFS 子图采样
3. 对 input 版本屏蔽 seed 的 text 和 comment 字段
4. ground_truth 保留完整原始子图

输出: benchmark_data.json，每个样本包含:
  - subgraph (屏蔽后的子图，含被屏蔽的中心节点)
  - ground_truth (仅中心节点被屏蔽字段的原始值)
"""

from __future__ import annotations

import json
import os
import glob
import random
import argparse
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# 候选中心节点类型 & 屏蔽字段
# ---------------------------------------------------------------------------

CANDIDATE_NODE_TYPES = {"Function", "Class"}

# 对每种节点类型，定义要屏蔽的字段
MASK_FIELDS: dict[str, list[str]] = {
    "Function": ["text", "comment"],
    "Class":    ["text", "comment"],
}

MASK_PLACEHOLDER = "[MASKED]"

# 需要跳过的节点（text 太短，没有补全价值）
MIN_TEXT_LENGTH = 50


# ---------------------------------------------------------------------------
# Graph loader — 参考 tools/VT/renderer.py
# ---------------------------------------------------------------------------

def load_graph(graph_path: str) -> dict:
    """加载 .graph.json 并建立索引。返回包含所有索引的字典。"""
    with open(graph_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    raw_nodes: list[dict] = data.get("nodes", [])
    raw_edges: list[dict] = data.get("edges", [])

    nodes_by_id: dict[int, dict] = {n["id"]: n for n in raw_nodes}

    # 构建邻接表 (无向) 和 contains 父子关系
    adj: dict[int, list[int]] = {nid: [] for nid in nodes_by_id}
    contains_parent: dict[int, int] = {}
    contains_children: dict[int, list[int]] = {nid: [] for nid in nodes_by_id}

    for e in raw_edges:
        src, tgt, etype = e["source"], e["target"], e["edgeType"]
        if tgt in nodes_by_id and src in nodes_by_id:
            adj[src].append(tgt)
            adj[tgt].append(src)
        if etype == "contains":
            if tgt not in contains_parent:
                contains_parent[tgt] = src
            contains_children.setdefault(src, []).append(tgt)

    return {
        "nodes_by_id": nodes_by_id,
        "edges": raw_edges,
        "adj": adj,
        "contains_parent": contains_parent,
        "contains_children": contains_children,
    }


# ---------------------------------------------------------------------------
# 子图采样 — 参考 CodeGraphRenderer.sample_subgraph()
# ---------------------------------------------------------------------------

def sample_subgraph(
    graph: dict,
    seed_id: int,
    k: int = 2,
    max_nodes: int = 150,
) -> tuple[list[dict], list[dict]]:
    """
    以 seed_id 为中心做两阶段子图采样。

    Phase 1: 沿 contains 边向上走到 Repo 根节点。
    Phase 2: k-hop BFS，优先 contains 边。

    Returns (sub_nodes, sub_edges)
    """
    nodes_by_id = graph["nodes_by_id"]
    adj = graph["adj"]
    contains_parent = graph["contains_parent"]
    contains_children = graph["contains_children"]
    edges = graph["edges"]

    if seed_id not in nodes_by_id:
        raise ValueError(f"Seed node {seed_id} not found")

    visited: set[int] = set()

    # Phase 1: ancestor chain upward
    cur = seed_id
    while cur is not None:
        visited.add(cur)
        cur = contains_parent.get(cur)

    # Phase 2: k-hop BFS, contains edges first
    queue: deque = deque()
    queue.append((seed_id, 0))
    bfs_seen: set[int] = {seed_id}

    while queue and len(visited) < max_nodes:
        node, dist = queue.popleft()
        if dist >= k:
            continue

        neighbours = list(adj.get(node, []))
        # 排序优先级: contains 子节点 > contains 父节点 > 其他
        neighbours.sort(
            key=lambda nb: (
                0 if nb in contains_children.get(node, []) else
                (1 if contains_parent.get(nb) == node else 2)
            )
        )

        for nb in neighbours:
            if nb in bfs_seen:
                continue
            if nb not in nodes_by_id:
                continue
            bfs_seen.add(nb)
            if len(visited) >= max_nodes:
                break
            visited.add(nb)
            queue.append((nb, dist + 1))

    # 过滤边
    sub_nodes = [nodes_by_id[nid] for nid in visited]
    sub_edges = [
        e for e in edges
        if e["source"] in visited and e["target"] in visited
    ]

    return sub_nodes, sub_edges


# ---------------------------------------------------------------------------
# 屏蔽逻辑
# ---------------------------------------------------------------------------

def _has_meaningful_content(value) -> bool:
    """判断字段值是否有实际内容（排除 null 占位符和空字符串）。"""
    if value is None:
        return False
    if isinstance(value, str):
        return value not in ("", "null")
    return True


def mask_node_fields(node: dict) -> dict:
    """
    返回 node 的拷贝，对中心节点屏蔽有实际内容的字段。

    只屏蔽真正有内容的字段: text 始终屏蔽，comment 仅当非 null/空时屏蔽。
    """
    masked = dict(node)
    nt = node.get("nodeType", "")
    if nt not in MASK_FIELDS:
        return masked

    for field in MASK_FIELDS[nt]:
        if field in masked and _has_meaningful_content(masked[field]):
            masked[field] = MASK_PLACEHOLDER
    return masked


# ---------------------------------------------------------------------------
# 种子节点筛选
# ---------------------------------------------------------------------------

def select_seeds(
    graph: dict,
    samples_per_graph: int = 4,
    min_text_len: int = MIN_TEXT_LENGTH,
) -> list[int]:
    """
    从图中随机选取合适的种子节点。

    条件:
    - 节点类型为 Function 或 Class
    - text 字段长度 >= min_text_len
    - 采样子图后至少有一定规模（在采样时验证）
    """
    nodes_by_id = graph["nodes_by_id"]
    candidates = []

    for nid, node in nodes_by_id.items():
        nt = node.get("nodeType", "")
        if nt not in CANDIDATE_NODE_TYPES:
            continue
        text = node.get("text", "")
        if isinstance(text, str) and len(text) >= min_text_len:
            candidates.append(nid)

    if len(candidates) <= samples_per_graph:
        return candidates

    return random.sample(candidates, samples_per_graph)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def prepare_benchmark_data(
    graph_dir: str,
    output_path: str,
    samples_per_graph: int = 4,
    k_hops: int = 2,
    max_nodes: int = 150,
    seed: int = 42,
    limit_graphs: Optional[int] = None,
) -> list[dict]:
    """
    遍历 graph_dir 下所有 .graph.json 文件，生成 benchmark 数据。
    """
    random.seed(seed)

    graph_files = sorted(glob.glob(os.path.join(graph_dir, "*.graph.json")))
    if not graph_files:
        raise ValueError(f"No .graph.json files found in {graph_dir}")

    if limit_graphs:
        graph_files = graph_files[:limit_graphs]

    all_samples = []
    skipped_empty = 0
    skipped_small_subgraph = 0

    for i, gf in enumerate(graph_files):
        graph_name = os.path.basename(gf)
        print(f"[{i+1}/{len(graph_files)}] Processing {graph_name} ...", end=" ", flush=True)

        try:
            graph = load_graph(gf)
        except Exception as e:
            print(f"SKIP (load error: {e})")
            continue

        # 选取种子节点
        seed_ids = select_seeds(graph, samples_per_graph=samples_per_graph)

        for sid in seed_ids:
            try:
                sub_nodes, sub_edges = sample_subgraph(
                    graph, sid, k=k_hops, max_nodes=max_nodes
                )
            except Exception as e:
                print(f"  seed={sid} SKIP (sample error: {e})")
                continue

            # 子图太小则跳过
            if len(sub_nodes) < 10:
                skipped_small_subgraph += 1
                continue

            seed_node = graph["nodes_by_id"][sid]
            seed_node_type = seed_node.get("nodeType", "?")

            # 构建输入（屏蔽 seed 的有意义字段）
            masked_seed = mask_node_fields(seed_node)
            # 记录实际被屏蔽的字段
            actually_masked = [
                f for f in MASK_FIELDS.get(seed_node_type, [])
                if masked_seed.get(f) == MASK_PLACEHOLDER
            ]

            input_nodes = []
            for n in sub_nodes:
                if n["id"] == sid:
                    input_nodes.append(masked_seed)
                else:
                    input_nodes.append(dict(n))

            sample = {
                "sample_id": f"{graph_name}::seed_{sid}",
                "source_graph": graph_name,
                "seed_node_id": sid,
                "seed_node_type": seed_node_type,
                "masked_fields": actually_masked,
                "subgraph": {
                    "nodes": input_nodes,
                    "edges": sub_edges,
                },
                "ground_truth": {
                    "node_id": sid,
                    **{field: seed_node.get(field) for field in actually_masked},
                },
            }
            all_samples.append(sample)

        print(f"done ({len(seed_ids)} seeds)")

    print()
    print(f"=== Summary ===")
    print(f"Total samples:     {len(all_samples)}")
    print(f"Graphs processed:  {len(graph_files)}")
    print(f"Skipped (small):   {skipped_small_subgraph}")
    print(f"Skipped (empty):   {skipped_empty}")

    # 写入输出
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_samples, fh, ensure_ascii=False, indent=2)

    print(f"Saved → {output_path}")
    return all_samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGM Benchmark Data Preparation")
    parser.add_argument(
        "--graph-dir",
        default="/home/10358884/10358884/tools/code_graph_data/swe-bench-lite",
        help="Path to swe-bench-lite .graph.json files",
    )
    parser.add_argument(
        "--output",
        default="/home/10358884/10358884/tools/benchmark/benchmark_data.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--samples-per-graph", type=int, default=4,
        help="Number of seed nodes per graph",
    )
    parser.add_argument(
        "--k-hops", type=int, default=2,
        help="BFS depth for subgraph expansion",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=150,
        help="Max nodes per subgraph",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--limit-graphs", type=int, default=None,
        help="Only process first N graphs (for testing)",
    )
    parser.add_argument(
        "--min-text-len", type=int, default=MIN_TEXT_LENGTH,
        help="Minimum text length for candidate seed nodes",
    )
    args = parser.parse_args()

    prepare_benchmark_data(
        graph_dir=args.graph_dir,
        output_path=args.output,
        samples_per_graph=args.samples_per_graph,
        k_hops=args.k_hops,
        max_nodes=args.max_nodes,
        seed=args.seed,
        limit_graphs=args.limit_graphs,
    )
