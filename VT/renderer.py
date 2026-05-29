"""
Code Graph Renderer — loads a .graph.json file, samples a subgraph
around a seed node, and renders it as a color-coded image.

Uses Graphviz (``dot`` engine) for layout and rendering:
- Rectangular nodes whose size adapts to the label text.
- Colours keyed by node / edge type with a built-in legend.
"""

import json
import os
import warnings
from collections import deque
from typing import Optional

import graphviz
import networkx as nx

# ---------------------------------------------------------------------------
# Color palettes (8 node types + 3 edge types, no overlap)
# ---------------------------------------------------------------------------

NODE_COLORS: dict[str, str] = {
    "Repo":      "#4C72B0",  # blue
    "Package":   "#DD8452",  # orange
    "File":      "#C44E52",  # red
    "TextFile":  "#937860",  # brown
    "Class":     "#55A868",  # green
    "Function":  "#8C6D31",  # olive
    "Attribute": "#8172B3",  # purple
    "Lambda":    "#64B5CD",  # steel blue
}

# 种子节点 (seed / masked node) 的专用高亮色 — 亮金色，与 8 种节点类型颜色不冲突
SEED_HIGHLIGHT_COLOR = "#FFD700"
SEED_BORDER_COLOR = "#CC0000"
SEED_PENWIDTH = "5.0"

EDGE_COLORS: dict[str, str] = {
    "contains": "#888888",  # gray — structural
    "calls":    "#CC3311",  # deep red-orange — behavioural
    "imports":  "#0077BB",  # deep blue — dependency
}

# Display order for the legend
NODE_TYPE_ORDER = [
    "Repo", "Package", "File", "TextFile",
    "Class", "Function", "Attribute", "Lambda",
]
EDGE_TYPE_ORDER = ["contains", "calls", "imports"]

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CodeGraphRenderer:
    """Load, sample, and render a code graph from a .graph.json file."""

    def __init__(self, graph_path: str) -> None:
        if not os.path.isfile(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        self.graph_path = graph_path
        self.graph_name = os.path.basename(graph_path)

        # Data structures populated by load_graph
        self.nodes_by_id: dict[int, dict] = {}
        self.edges: list[dict] = []
        self.adj: dict[int, list[int]] = {}            # undirected neighbour index
        self.contains_parent: dict[int, int] = {}       # node → parent via contains
        self.contains_children: dict[int, list[int]] = {}  # node → children via contains

        self.load_graph()

    # ------------------------------------------------------------------
    # 1. Data loader
    # ------------------------------------------------------------------

    def load_graph(self) -> None:
        """Parse the .graph.json file and build in-memory indexes."""
        with open(self.graph_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        raw_nodes: list[dict] = data.get("nodes", [])
        raw_edges: list[dict] = data.get("edges", [])

        self.nodes_by_id = {n["id"]: n for n in raw_nodes}

        self.adj = {}
        self.contains_parent = {}
        self.contains_children = {}

        for nid in self.nodes_by_id:
            self.adj[nid] = []
            self.contains_children[nid] = []

        for e in raw_edges:
            src, tgt, etype = e["source"], e["target"], e["edgeType"]

            if tgt in self.nodes_by_id and src in self.nodes_by_id:
                self.adj[src].append(tgt)
                self.adj[tgt].append(src)

            if etype == "contains":
                if tgt not in self.contains_parent:
                    self.contains_parent[tgt] = src
                self.contains_children.setdefault(src, []).append(tgt)

        self.edges = raw_edges

    # ------------------------------------------------------------------
    # 2. Node label helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _natural_identifier(node: dict) -> str:
        """Return the human-readable identifier for a node (without ID prefix)."""
        nt = node.get("nodeType", "")

        if nt == "Repo":
            return node.get("repoName", "?")
        elif nt == "Package":
            return node.get("name", "?")
        elif nt == "File":
            fp = node.get("filePath", "")
            fn = node.get("fileName", "")
            return f"{fp}/{fn}" if fp else fn
        elif nt == "TextFile":
            p = node.get("path", "")
            n = node.get("name", "")
            return f"{p}/{n}" if p else n
        elif nt == "Class":
            return node.get("className", "?")
        elif nt == "Function":
            return node.get("name", "?")
        elif nt == "Attribute":
            return node.get("name", "?")
        elif nt == "Lambda":
            return f"lambda@{node['id']}"
        return f"<{nt}>"

    @staticmethod
    def _node_label(node: dict) -> str:
        """Graphviz HTML-style label: actual numeric ID + natural identifier.

        Returns a string like ``<<B>327: Flask</B>>`` so Graphviz auto-sizes
        the bounding rectangle to the text width.
        """
        raw = f"{node['id']}: {CodeGraphRenderer._natural_identifier(node)}"
        if len(raw) > 60:
            raw = raw[:57] + "..."
        return raw

    # ------------------------------------------------------------------
    # 3. Subgraph sampler
    # ------------------------------------------------------------------

    def sample_subgraph(
        self, seed_id: int, k: int = 2, max_nodes: int = 200
    ) -> tuple[nx.DiGraph, dict[int, str]]:
        """
        Two-phase sampling around *seed_id*:

        Phase 1 — walk up ``contains`` edges to the Repo root (always included).
        Phase 2 — k-hop BFS from *seed_id*, prioritising contains edges.
                   Stops at *max_nodes*.

        Returns a ``(DiGraph, labels_dict)`` tuple.
        """
        if seed_id not in self.nodes_by_id:
            raise ValueError(f"Seed node {seed_id} not found in graph")

        visited: set[int] = set()

        # Phase 1: ancestor chain
        cur = seed_id
        while cur is not None:
            visited.add(cur)
            cur = self.contains_parent.get(cur)

        # Phase 2: k-hop BFS, contains edges first
        queue: deque = deque()
        queue.append((seed_id, 0))
        bfs_seen: set[int] = {seed_id}

        while queue and len(visited) < max_nodes:
            node, dist = queue.popleft()
            if dist >= k:
                continue

            neighbours = list(self.adj.get(node, []))
            neighbours.sort(key=lambda nb: 0 if nb in self.contains_children.get(node, []) else
                          (1 if self.contains_parent.get(nb) == node else 2))

            for nb in neighbours:
                if nb in bfs_seen:
                    continue
                if nb not in self.nodes_by_id:
                    continue
                bfs_seen.add(nb)
                if len(visited) >= max_nodes:
                    break
                visited.add(nb)
                queue.append((nb, dist + 1))

        # Phase 3: filter edges where both ends are in visited
        sub_nodes = visited
        sub_edges: list[dict] = [
            e for e in self.edges
            if e["source"] in sub_nodes and e["target"] in sub_nodes
        ]

        G = nx.DiGraph()
        for nid in sub_nodes:
            node = self.nodes_by_id[nid]
            G.add_node(nid, **node)
        for e in sub_edges:
            G.add_edge(e["source"], e["target"], edgeType=e["edgeType"])

        labels = {nid: self._node_label(self.nodes_by_id[nid]) for nid in sub_nodes}

        return G, labels

    # ------------------------------------------------------------------
    # 4. Graphviz-based render
    # ------------------------------------------------------------------

    def _build_legend_html(self, present_node_types: set[str],
                           present_edge_types: set[str]) -> str:
        """Build an HTML table string used as a Graphviz node label for the legend."""
        rows: list[str] = []
        rows.append('<TR><TD COLSPAN="2" ALIGN="CENTER"><B>Legend</B></TD></TR>')

        # 种子节点高亮标识（放在最前面）
        rows.append(
            f'<TR><TD BGCOLOR="{SEED_HIGHLIGHT_COLOR}" WIDTH="20" HEIGHT="12">'
            f'<FONT COLOR="{SEED_BORDER_COLOR}">&#9632;</FONT></TD>'
            f'<TD ALIGN="LEFT"><B>[MASKED] Target Node</B></TD></TR>'
        )

        for nt in NODE_TYPE_ORDER:
            if nt in present_node_types:
                color = NODE_COLORS[nt]
                rows.append(
                    f'<TR><TD BGCOLOR="{color}" WIDTH="20" HEIGHT="12"> </TD>'
                    f'<TD ALIGN="LEFT">{nt}</TD></TR>'
                )

        # Separator row
        if present_node_types and present_edge_types:
            rows.append(
                '<TR><TD COLSPAN="2"><FONT POINT-SIZE="1"> </FONT></TD></TR>'
            )

        for et in EDGE_TYPE_ORDER:
            if et in present_edge_types:
                color = EDGE_COLORS[et]
                rows.append(
                    f'<TR><TD><FONT COLOR="{color}" POINT-SIZE="16">—</FONT></TD>'
                    f'<TD ALIGN="LEFT">{et}</TD></TR>'
                )

        return (
            '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2" '
            'CELLPADDING="3" BGCOLOR="white">\n'
            + "\n".join(rows)
            + "\n</TABLE>>"
        )

    def _build_digraph(self, G: nx.DiGraph, labels: dict[int, str],
                       seed_id: int, figsize: tuple[float, float] = (24, 18),
                       dpi: int = 150) -> graphviz.Digraph:
        """Build a :class:`graphviz.Digraph` from the subgraph."""
        # Collect type metadata
        node_type_map: dict[int, str] = {}
        present_node_types: set[str] = set()
        for n, data in G.nodes(data=True):
            nt = data.get("nodeType", "?")
            node_type_map[n] = nt
            present_node_types.add(nt)

        edge_type_map: dict[tuple[int, int], str] = {}
        present_edge_types: set[str] = set()
        for u, v, data in G.edges(data=True):
            et = data.get("edgeType", "?")
            edge_type_map[(u, v)] = et
            present_edge_types.add(et)

        # Create Digraph
        dot = graphviz.Digraph(
            name="codegraph",
            comment=f"Code Graph: {self.graph_name}  —  seed={seed_id}",
            format="png",
            engine="dot",
        )
        dot.attr(
            rankdir="TB",
            newrank="true",
            bgcolor="white",
            fontname="Helvetica",
            fontcolor="black",
            label=(
                f"Code Graph: {os.path.splitext(self.graph_name)[0]}\\n"
                f"seed={seed_id}"
            ),
            labelloc="t",
            labeljust="c",
            fontsize="16",
            size=f"{figsize[0]},{figsize[1]}!",
            ratio="fill",
        )
        dot.graph_attr["dpi"] = str(dpi)
        dot.attr("node",
                 shape="box", style="filled,rounded",
                 fontname="Helvetica", fontsize="10",
                 fontcolor="black", color="#333333", penwidth="1.2")
        dot.attr("edge",
                 fontname="Helvetica", fontsize="8",
                 fontcolor="black")

        # --- Legend: fixed to top-left, larger and more readable ---
        legend_label = self._build_legend_html(present_node_types, present_edge_types)

        # 使用独立 subgraph + rank=source 将图例固定在顶部
        with dot.subgraph(name="cluster_legend_top") as legend_sub:
            legend_sub.attr(rank="source", style="invis")
            dot.node("legend", label=legend_label,
                     shape="box", style="filled,rounded",
                     fillcolor="#FFFEF5", color="#888888",
                     fontsize="12", margin="0.3,0.2",
                     fontname="Helvetica")

        # 找到 Repo 根节点，用不可见边将图例固定在根节点左侧（同一 rank）
        root_nid = None
        for n in G.nodes():
            if node_type_map.get(n) == "Repo":
                root_nid = str(n)
                break
        if root_nid:
            dot.edge("legend", root_nid, style="invis", weight="0")

        # Seed highlight node (invisible — just for the legend to connect to)
        # Highlighting: seed gets bold label + thicker border
        for n in G.nodes():
            nid_str = str(n)
            lbl = labels.get(n, str(n))

            fillcolor = NODE_COLORS.get(node_type_map.get(n, "?"), "#AAAAAA")
            extras: dict[str, str] = {
                "fillcolor": fillcolor,
                "fontsize": "10",
            }

            if n == seed_id:
                # 种子节点：亮金色填充 + 红色粗边框 + [MASKED] 标记
                extras["fillcolor"] = SEED_HIGHLIGHT_COLOR
                extras["color"] = SEED_BORDER_COLOR
                extras["penwidth"] = SEED_PENWIDTH
                extras["fontsize"] = "12"
                html_label = f'<<B><FONT COLOR="#CC0000">[MASKED]</FONT><BR/>{_escape_html(lbl)}</B>>'
            else:
                html_label = f'<<B>{_escape_html(lbl)}</B>>'

            dot.node(nid_str, label=html_label, **extras)

        # Edges grouped by type for color
        for et in EDGE_TYPE_ORDER:
            e_list = [(u, v) for (u, v), t in edge_type_map.items() if t == et]
            if not e_list:
                continue
            edge_color = EDGE_COLORS[et]
            for u, v in e_list:
                dot.edge(str(u), str(v),
                         label=f" {et}",
                         color=edge_color, fontcolor=edge_color)

        return dot

    def render(
        self,
        seed_id: int,
        k: int = 2,
        max_nodes: int = 200,
        output_path: Optional[str] = None,
        show: bool = True,
        figsize: tuple[float, float] = (24, 18),
        dpi: int = 150,
    ) -> str:
        """
        Sample a subgraph around *seed_id* and render it to an image via Graphviz.

        Parameters
        ----------
        seed_id : int
            The node id to centre the subgraph on.
        k : int
            BFS depth (default 2).
        max_nodes : int
            Hard cap on the number of nodes in the subgraph.
        output_path : str or None
            Where to save the PNG.  If None, auto-generates a path under
            ``tools/VT/outputs/``.
        show : bool
            If True, attempt to open the image with the system viewer.
        figsize : tuple
            Output canvas size in inches (width, height).  Mapped to
            Graphviz ``size`` attribute with ``ratio="fill"``.
        dpi : int
            Output resolution.  Set as Graphviz ``dpi`` graph attribute.
            The final pixel dimensions are ``figsize * dpi``.

        Returns
        -------
        str
            The path where the image was saved.
        """
        G, labels = self.sample_subgraph(seed_id, k=k, max_nodes=max_nodes)

        if G.number_of_nodes() <= 1:
            raise ValueError("Subgraph is empty or singleton — nothing to render.")

        dot = self._build_digraph(G, labels, seed_id, figsize=figsize, dpi=dpi)

        # Determine output path
        if output_path is None:
            out_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(self.graph_name)[0]
            output_path = os.path.join(out_dir, f"{stem}_seed{seed_id}_k{k}")

        # graphviz.render expects *filename* without extension
        base, ext = os.path.splitext(output_path)
        if ext and ext.lower() in (".png", ".svg", ".pdf"):
            output_path_noext = base
        else:
            output_path_noext = output_path
            output_path = output_path_noext + ".png"

        dot.render(filename=output_path_noext, cleanup=True)
        print(f"[CodeGraphRenderer] Saved → {output_path}")

        if show:
            dot.view()

        return output_path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _escape_html(text: str) -> str:
    """Minimal XML-escaping for text placed inside HTML labels."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
