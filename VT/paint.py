import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ========== 构建树 ==========
G = nx.DiGraph()
edges = [
    ("CEO", "技术VP"),
    ("CEO", "运营VP"),
    ("CEO", "财务VP"),
    ("技术VP", "前端部"),
    ("技术VP", "后端部"),
    ("技术VP", "AI部"),
    ("运营VP", "市场部"),
    ("运营VP", "客服部"),
    ("财务VP", "会计部"),
    ("财务VP", "审计部"),
    ("前端部", "Web组"),
    ("前端部", "移动组"),
    ("后端部", "Java组"),
    ("后端部", "Go组"),
    ("AI部", "CV组"),
    ("AI部", "NLP组"),
    ("市场部", "品牌组"),
    ("市场部", "增长组"),
]
G.add_edges_from(edges)

# ========== 手动计算层次布局 ==========
def tree_layout(G, root, x_gap=2.0, y_gap=1.5):
    """
    自底向上分配 x 坐标，自顶向下分配 y 坐标
    保证叶子节点均匀分布，整棵树水平展开
    """
    depth = nx.shortest_path_length(G, root)
    max_depth = max(depth.values())

    children = {n: list(G.successors(n)) for n in G.nodes()}
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]

    # 叶子节点从左到右均匀排列
    leaf_x = {}
    for i, leaf in enumerate(leaves):
        leaf_x[leaf] = i * x_gap

    # 自底向上：父节点 x = 子节点 x 的中心
    pos = {}
    for d in range(max_depth, -1, -1):
        level_nodes = [n for n in G.nodes() if depth[n] == d]
        for n in level_nodes:
            if n in leaf_x:
                pos[n] = (leaf_x[n], -depth[n] * y_gap)
            else:
                child_xs = [pos[c][0] for c in children[n]]
                pos[n] = (sum(child_xs) / len(child_xs), -depth[n] * y_gap)

    return pos

pos = tree_layout(G, root="CEO", x_gap=2.0, y_gap=1.5)

# ========== 颜色方案 ==========
depth = nx.shortest_path_length(G, "CEO")
max_depth = max(depth.values())

level_colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]
node_colors = [level_colors[depth[n]] for n in G.nodes()]
edge_colors = [level_colors[depth[u]] for u, v in G.edges()]
node_sizes = [1600 - depth[n] * 300 for n in G.nodes()]

# ========== 绘图 ==========
fig, ax = plt.subplots(figsize=(20, 10), facecolor="#FAFAFA")
ax.set_facecolor("#FAFAFA")

# 边：轻微弧线
nx.draw_networkx_edges(
    G, pos, ax=ax,
    edge_color=edge_colors,
    width=2.5,
    alpha=0.7,
    connectionstyle="arc3,rad=0.1",
    arrows=True,
    arrowsize=18,
    arrowstyle="-|>",
    min_source_margin=20,
    min_target_margin=20,
)

# 节点
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    edgecolors="white",
    linewidths=2,
    alpha=0.95,
)

# 标签
nx.draw_networkx_labels(
    G, pos, ax=ax,
    font_size=11,
    font_weight="bold",
)

# 图例
legend_handles = []
for d in range(max_depth + 1):
    label = ["第0层(根)", "第1层", "第2层", "第3层"][d] if d <= 3 else f"第{d}层"
    patch = mpatches.Patch(color=level_colors[d], label=label)
    legend_handles.append(patch)

ax.legend(
    handles=legend_handles,
    loc="upper right", fontsize=10,
    frameon=True, facecolor="white", edgecolor="#CCCCCC",
    title="层级", title_fontsize=11,
)

ax.set_title("树状结构图 — 层次展开布局", fontsize=18, fontweight="bold", pad=20)
ax.axis("off")
plt.tight_layout()
plt.savefig("tree_graph.png", dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
plt.show()


