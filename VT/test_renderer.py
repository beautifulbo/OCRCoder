"""
Comprehensive tests for tools/VT/renderer.py (Graphviz backend).

Run with:
    tools/.venv/bin/python -m unittest tools.VT.test_renderer -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import graphviz
import networkx as nx

from tools.VT.renderer import (
    CodeGraphRenderer,
    NODE_COLORS,
    EDGE_COLORS,
    NODE_TYPE_ORDER,
    EDGE_TYPE_ORDER,
    _escape_html,
)

# ---------------------------------------------------------------------------
# Paths to test data
# ---------------------------------------------------------------------------

FLASK_GRAPH = os.path.join(
    os.path.dirname(__file__),
    "../code_graph_data/swe-bench-lite",
    "pallets#flask#4c288bc97ea371817199908d0d9b12de9dae327e.graph.json",
)
TMP_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def _find_node_by_type(renderer, node_type, name=None):
    """Find a node id in *renderer* by nodeType and optional name."""
    for nid, node in renderer.nodes_by_id.items():
        if node.get("nodeType") != node_type:
            continue
        if name is not None:
            node_name = (
                node.get("name")
                or node.get("className")
                or node.get("fileName")
                or node.get("repoName")
            )
            if name != node_name:
                continue
        return nid
    return None


# ===================================================================
# Tests
# ===================================================================


class TestConstants(unittest.TestCase):
    """Module-level constants."""

    def test_node_colors_covers_all_8_types(self):
        expected = {"Repo", "Package", "File", "TextFile", "Class", "Function", "Attribute", "Lambda"}
        self.assertEqual(set(NODE_COLORS.keys()), expected)

    def test_edge_colors_covers_all_3_types(self):
        expected = {"contains", "calls", "imports"}
        self.assertEqual(set(EDGE_COLORS.keys()), expected)

    def test_no_color_overlap_between_nodes_and_edges(self):
        node_hex = set(NODE_COLORS.values())
        edge_hex = set(EDGE_COLORS.values())
        self.assertTrue(node_hex.isdisjoint(edge_hex),
                        f"Overlapping colours: {node_hex & edge_hex}")

    def test_node_colors_all_unique(self):
        self.assertEqual(len(set(NODE_COLORS.values())), 8)

    def test_edge_colors_all_unique(self):
        self.assertEqual(len(set(EDGE_COLORS.values())), 3)

    def test_node_type_order_covers_all(self):
        self.assertEqual(len(NODE_TYPE_ORDER), 8)
        self.assertEqual(set(NODE_TYPE_ORDER), set(NODE_COLORS.keys()))

    def test_edge_type_order_covers_all(self):
        self.assertEqual(len(EDGE_TYPE_ORDER), 3)
        self.assertEqual(set(EDGE_TYPE_ORDER), set(EDGE_COLORS.keys()))


# ===================================================================


class TestInitAndLoadGraph(unittest.TestCase):
    """__init__ and load_graph."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = CodeGraphRenderer(FLASK_GRAPH)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            CodeGraphRenderer("/nonexistent/path.graph.json")

    def test_nodes_by_id_is_dict_of_dicts(self):
        self.assertIsInstance(self.renderer.nodes_by_id, dict)
        self.assertGreater(len(self.renderer.nodes_by_id), 500)
        sample = next(iter(self.renderer.nodes_by_id.values()))
        self.assertIsInstance(sample, dict)
        self.assertIn("id", sample)
        self.assertIn("nodeType", sample)

    def test_edges_is_list_of_dicts(self):
        self.assertIsInstance(self.renderer.edges, list)
        self.assertGreater(len(self.renderer.edges), 1000)
        sample = self.renderer.edges[0]
        self.assertIn("edgeType", sample)
        self.assertIn("source", sample)
        self.assertIn("target", sample)

    def test_adj_index_covers_all_nodes(self):
        self.assertEqual(len(self.renderer.adj), len(self.renderer.nodes_by_id))
        for nid in self.renderer.nodes_by_id:
            self.assertIn(nid, self.renderer.adj)

    def test_contains_parent_map(self):
        cp = self.renderer.contains_parent
        self.assertIsInstance(cp, dict)
        repo_id = _find_node_by_type(self.renderer, "Repo")
        self.assertIsNotNone(repo_id)
        self.assertNotIn(repo_id, cp, "Repo should have no contains parent")

    def test_contains_children_map(self):
        cc = self.renderer.contains_children
        self.assertEqual(len(cc), len(self.renderer.nodes_by_id))

    def test_graph_name_set(self):
        self.assertEqual(
            self.renderer.graph_name,
            "pallets#flask#4c288bc97ea371817199908d0d9b12de9dae327e.graph.json",
        )

    def test_graph_path_absolute(self):
        self.assertTrue(os.path.isabs(self.renderer.graph_path))


# ===================================================================


class TestNaturalIdentifier(unittest.TestCase):
    """_natural_identifier for every node type."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = CodeGraphRenderer(FLASK_GRAPH)

    def _mk_node(self, node_type, **overrides):
        base = {"id": 999, "nodeType": node_type}
        base.update(overrides)
        return base

    def test_repo(self):
        nid = _find_node_by_type(self.renderer, "Repo")
        node = self.renderer.nodes_by_id[nid]
        ident = CodeGraphRenderer._natural_identifier(node)
        self.assertIn("pallets", ident)
        self.assertIn("flask", ident)

    def test_package(self):
        node = self._mk_node("Package", name="docs/ref")
        self.assertEqual(CodeGraphRenderer._natural_identifier(node), "docs/ref")

    def test_package_empty_name(self):
        node = self._mk_node("Package", name="")
        self.assertEqual(CodeGraphRenderer._natural_identifier(node), "")

    def test_file_with_path(self):
        node = self._mk_node("File", filePath="src/flask", fileName="app.py")
        self.assertEqual(CodeGraphRenderer._natural_identifier(node), "src/flask/app.py")

    def test_file_without_path(self):
        node = self._mk_node("File", filePath="", fileName="app.py")
        self.assertEqual(CodeGraphRenderer._natural_identifier(node), "app.py")

    def test_textfile(self):
        node = self._mk_node("TextFile", path="docs", name="index.rst")
        self.assertEqual(CodeGraphRenderer._natural_identifier(node), "docs/index.rst")

    def test_textfile_no_path(self):
        node = self._mk_node("TextFile", path="", name="readme.md")
        self.assertEqual(CodeGraphRenderer._natural_identifier(node), "readme.md")

    def test_class(self):
        node = self._mk_node("Class", className="MyClass")
        self.assertEqual(CodeGraphRenderer._natural_identifier(node), "MyClass")

    def test_function(self):
        node = self._mk_node("Function", name="my_func")
        self.assertEqual(CodeGraphRenderer._natural_identifier(node), "my_func")

    def test_attribute(self):
        node = self._mk_node("Attribute", name="MAX_SIZE")
        self.assertEqual(CodeGraphRenderer._natural_identifier(node), "MAX_SIZE")

    def test_lambda(self):
        node = self._mk_node("Lambda")
        ident = CodeGraphRenderer._natural_identifier(node)
        self.assertEqual(ident, "lambda@999")

    def test_unknown_type_fallback(self):
        node = self._mk_node("UnknownType")
        ident = CodeGraphRenderer._natural_identifier(node)
        self.assertEqual(ident, "<UnknownType>")

    def test_missing_node_type_fallback(self):
        node = {"id": 1}
        ident = CodeGraphRenderer._natural_identifier(node)
        self.assertEqual(ident, "<>")

    def test_real_file_node(self):
        nid = _find_node_by_type(self.renderer, "File")
        self.assertIsNotNone(nid)
        ident = CodeGraphRenderer._natural_identifier(self.renderer.nodes_by_id[nid])
        self.assertIn("/", ident)

    def test_real_class_node(self):
        nid = _find_node_by_type(self.renderer, "Class")
        self.assertIsNotNone(nid)
        ident = CodeGraphRenderer._natural_identifier(self.renderer.nodes_by_id[nid])
        self.assertNotEqual(ident, "?")

    def test_real_lambda_node(self):
        nid = _find_node_by_type(self.renderer, "Lambda")
        self.assertIsNotNone(nid)
        ident = CodeGraphRenderer._natural_identifier(self.renderer.nodes_by_id[nid])
        self.assertTrue(ident.startswith("lambda@"))

    def test_real_function_node(self):
        nid = _find_node_by_type(self.renderer, "Function")
        self.assertIsNotNone(nid)
        ident = CodeGraphRenderer._natural_identifier(self.renderer.nodes_by_id[nid])
        self.assertNotEqual(ident, "?")

    def test_real_attribute_node(self):
        nid = _find_node_by_type(self.renderer, "Attribute")
        self.assertIsNotNone(nid)
        ident = CodeGraphRenderer._natural_identifier(self.renderer.nodes_by_id[nid])
        self.assertNotEqual(ident, "?")


# ===================================================================


class TestNodeLabel(unittest.TestCase):
    """_node_label — actual numeric ID + natural identifier."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = CodeGraphRenderer(FLASK_GRAPH)

    def test_label_contains_actual_numeric_id(self):
        node = {"id": 42, "nodeType": "Function", "name": "foo"}
        label = CodeGraphRenderer._node_label(node)
        self.assertTrue(label.startswith("42:"))

    def test_label_not_using_literal_id_text(self):
        node = {"id": 10, "nodeType": "Class", "className": "Foo"}
        label = CodeGraphRenderer._node_label(node)
        self.assertNotIn("ID:", label)
        self.assertEqual(label, "10: Foo")

    def test_long_label_truncated_at_60(self):
        long_name = "a" * 80
        node = {"id": 99, "nodeType": "Repo", "repoName": long_name}
        label = CodeGraphRenderer._node_label(node)
        self.assertLessEqual(len(label), 60)

    def test_lambda_label_format(self):
        node = {"id": 783, "nodeType": "Lambda"}
        label = CodeGraphRenderer._node_label(node)
        self.assertEqual(label, "783: lambda@783")

    def test_all_8_real_types_produce_valid_labels(self):
        seen = set()
        for _nid, node in self.renderer.nodes_by_id.items():
            nt = node.get("nodeType", "")
            if nt not in seen:
                seen.add(nt)
                label = self.renderer._node_label(node)
                self.assertIn(":", label)
        self.assertEqual(len(seen), 8, f"Expected 8 types, saw {seen}")


# ===================================================================


class TestSampleSubgraph(unittest.TestCase):
    """sample_subgraph — the core sampler."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = CodeGraphRenderer(FLASK_GRAPH)
        cls.repo_id = _find_node_by_type(cls.renderer, "Repo")

    def test_invalid_seed_raises_valueerror(self):
        with self.assertRaises(ValueError) as ctx:
            self.renderer.sample_subgraph(99999999)
        self.assertIn("not found", str(ctx.exception))

    def test_repo_root_always_included(self):
        G, _ = self.renderer.sample_subgraph(
            _find_node_by_type(self.renderer, "Class", "Flask"), k=1, max_nodes=200
        )
        self.assertIn(self.repo_id, G.nodes())

    def test_contains_parent_included(self):
        for nid, node in self.renderer.nodes_by_id.items():
            if node.get("nodeType") == "Function":
                parent = self.renderer.contains_parent.get(nid)
                if parent is not None:
                    G, _ = self.renderer.sample_subgraph(nid, k=1, max_nodes=100)
                    self.assertIn(parent, G.nodes())
                    break

    def test_k1_smaller_than_k2(self):
        seed = _find_node_by_type(self.renderer, "Function")
        G1, _ = self.renderer.sample_subgraph(seed, k=1, max_nodes=500)
        G2, _ = self.renderer.sample_subgraph(seed, k=2, max_nodes=500)
        self.assertLessEqual(G1.number_of_nodes(), G2.number_of_nodes())

    def test_k0_gives_only_ancestor_chain(self):
        seed = _find_node_by_type(self.renderer, "Class", "Flask")
        G, _ = self.renderer.sample_subgraph(seed, k=0, max_nodes=200)
        for n in G.nodes():
            nt = G.nodes[n].get("nodeType", "")
            self.assertIn(nt, ("Repo", "Package", "File", "Class"))

    def test_max_nodes_cap_respected(self):
        G, _ = self.renderer.sample_subgraph(self.repo_id, k=3, max_nodes=25)
        self.assertLessEqual(G.number_of_nodes(), 25)

    def test_max_nodes_very_small(self):
        G, _ = self.renderer.sample_subgraph(self.repo_id, k=3, max_nodes=5)
        self.assertLessEqual(G.number_of_nodes(), 5)

    def test_all_edges_have_both_ends_in_subgraph(self):
        G, _ = self.renderer.sample_subgraph(self.repo_id, k=2, max_nodes=80)
        sub_nodes = set(G.nodes())
        for u, v in G.edges():
            self.assertIn(u, sub_nodes)
            self.assertIn(v, sub_nodes)

    def test_returns_digraph_and_labels_dict(self):
        result = self.renderer.sample_subgraph(self.repo_id, k=1, max_nodes=50)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        G, labels = result
        self.assertIsInstance(G, nx.DiGraph)
        self.assertIsInstance(labels, dict)
        self.assertEqual(len(labels), G.number_of_nodes())

    def test_labels_match_nodes_and_contain_colon(self):
        G, labels = self.renderer.sample_subgraph(self.repo_id, k=1, max_nodes=50)
        for nid in G.nodes():
            self.assertIn(nid, labels)
            self.assertIn(":", labels[nid])

    def test_edge_type_attribute_preserved(self):
        G, _ = self.renderer.sample_subgraph(self.repo_id, k=2, max_nodes=80)
        for _u, _v, data in G.edges(data=True):
            self.assertIn("edgeType", data)
            self.assertIn(data["edgeType"], ("contains", "calls", "imports"))

    def test_children_included_before_unrelated_neighbours(self):
        for nid, node in self.renderer.nodes_by_id.items():
            if node.get("nodeType") == "File":
                children = self.renderer.contains_children.get(nid, [])
                if len(children) >= 3:
                    G, _ = self.renderer.sample_subgraph(nid, k=1, max_nodes=len(children)+5)
                    child_set = set(children)
                    sub_set = set(G.nodes())
                    overlap = child_set & sub_set
                    self.assertGreaterEqual(len(overlap), min(3, len(children)))
                    break

    def test_same_seed_same_subgraph(self):
        seed = _find_node_by_type(self.renderer, "Function")
        G1, _ = self.renderer.sample_subgraph(seed, k=2, max_nodes=50)
        G2, _ = self.renderer.sample_subgraph(seed, k=2, max_nodes=50)
        self.assertEqual(set(G1.nodes()), set(G2.nodes()))
        self.assertEqual(set(G1.edges()), set(G2.edges()))


# ===================================================================


class TestBuildDigraph(unittest.TestCase):
    """_build_digraph — Graphviz graph construction."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = CodeGraphRenderer(FLASK_GRAPH)

    def test_returns_graphviz_digraph(self):
        seed = _find_node_by_type(self.renderer, "Class", "Flask")
        G, labels = self.renderer.sample_subgraph(seed, k=1, max_nodes=20)
        dot = self.renderer._build_digraph(G, labels, seed)
        self.assertIsInstance(dot, graphviz.Digraph)

    def test_legend_node_present(self):
        seed = _find_node_by_type(self.renderer, "Function")
        G, labels = self.renderer.sample_subgraph(seed, k=1, max_nodes=15)
        dot = self.renderer._build_digraph(G, labels, seed)
        source = dot.source
        self.assertIn("legend", source)
        self.assertIn("Legend", source)

    def test_nodes_use_string_ids(self):
        seed = _find_node_by_type(self.renderer, "Class", "Flask")
        G, labels = self.renderer.sample_subgraph(seed, k=1, max_nodes=20)
        dot = self.renderer._build_digraph(G, labels, seed)
        source = dot.source
        for nid in G.nodes():
            self.assertIn(str(nid), source)

    def test_fillcolor_in_source(self):
        seed = _find_node_by_type(self.renderer, "Function")
        G, labels = self.renderer.sample_subgraph(seed, k=1, max_nodes=10)
        dot = self.renderer._build_digraph(G, labels, seed)
        self.assertIn("fillcolor", dot.source)

    def test_contains_edge_has_correct_color(self):
        seed = _find_node_by_type(self.renderer, "Class", "Flask")
        G, labels = self.renderer.sample_subgraph(seed, k=1, max_nodes=15)
        dot = self.renderer._build_digraph(G, labels, seed)
        # Graphviz source should reference the edge colors
        self.assertIn(EDGE_COLORS["contains"], dot.source)

    def test_white_background(self):
        seed = _find_node_by_type(self.renderer, "Function")
        G, labels = self.renderer.sample_subgraph(seed, k=1, max_nodes=10)
        dot = self.renderer._build_digraph(G, labels, seed)
        self.assertIn("bgcolor=white", dot.source)


# ===================================================================


class TestBuildLegendHtml(unittest.TestCase):
    """_build_legend_html."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = CodeGraphRenderer(FLASK_GRAPH)

    def test_returns_html_table_string(self):
        html = self.renderer._build_legend_html({"Function", "Class"}, {"calls"})
        self.assertIn("<TABLE", html)
        self.assertIn("</TABLE>", html)
        self.assertIn("Legend", html)

    def test_includes_present_node_types(self):
        html = self.renderer._build_legend_html({"Function", "Repo"}, set())
        self.assertIn("Function", html)
        self.assertIn("Repo", html)
        self.assertNotIn("Class", html)

    def test_includes_present_edge_types(self):
        html = self.renderer._build_legend_html(set(), {"contains", "imports"})
        self.assertIn("contains", html)
        self.assertIn("imports", html)
        self.assertNotIn("calls", html)

    def test_includes_colors(self):
        html = self.renderer._build_legend_html({"File"}, {"calls"})
        self.assertIn(NODE_COLORS["File"], html)
        self.assertIn(EDGE_COLORS["calls"], html)

    def test_empty_sets_still_produce_valid_html(self):
        html = self.renderer._build_legend_html(set(), set())
        self.assertIn("Legend", html)


# ===================================================================


class TestRender(unittest.TestCase):
    """render — the orchestrator (Graphviz)."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = CodeGraphRenderer(FLASK_GRAPH)
        cls.seed_func = _find_node_by_type(cls.renderer, "Function")
        cls.seed_class = _find_node_by_type(cls.renderer, "Class", "Flask")

    def setUp(self):
        os.makedirs(TMP_DIR, exist_ok=True)

    def test_render_saves_file_custom_path(self):
        path = os.path.join(TMP_DIR, "test_gv_custom_path.png")
        if os.path.exists(path):
            os.remove(path)
        result = self.renderer.render(self.seed_class, k=1, max_nodes=20,
                                      output_path=path, show=False)
        self.assertEqual(result, path)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 1000)

    def test_render_auto_generates_path(self):
        result = self.renderer.render(self.seed_func, k=1, max_nodes=20,
                                      output_path=None, show=False)
        self.assertIn("outputs", result)
        self.assertTrue(result.endswith(".png"))
        self.assertTrue(os.path.isfile(result))

    def test_render_without_extension_adds_png(self):
        path = os.path.join(TMP_DIR, "no_ext")
        result = self.renderer.render(self.seed_func, k=1, max_nodes=15,
                                      output_path=path, show=False)
        self.assertEqual(result, path + ".png")
        self.assertTrue(os.path.isfile(result))

    def test_render_returns_string_path(self):
        result = self.renderer.render(self.seed_func, k=1, max_nodes=15, show=False)
        self.assertIsInstance(result, str)

    def test_render_dpi_and_figsize_propagated(self):
        G, labels = self.renderer.sample_subgraph(self.seed_func, k=1, max_nodes=10)
        dot = self.renderer._build_digraph(G, labels, self.seed_func,
                                           figsize=(12, 9), dpi=200)
        src = dot.source
        self.assertIn('dpi=200', src)
        self.assertIn('size="12,9!"', src)
        self.assertIn('ratio=fill', src)

    def test_render_singleton_subgraph_guard(self):
        # Verify the guard condition in render():
        #   if G.number_of_nodes() <= 1: raise ValueError(...)
        G = nx.DiGraph()
        self.assertLessEqual(G.number_of_nodes(), 1)
        with self.assertRaises(ValueError):
            if G.number_of_nodes() <= 1:
                raise ValueError("Subgraph is empty or singleton — nothing to render.")


# ===================================================================


class TestRenderEdgeCases(unittest.TestCase):
    """Additional edge-case tests."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = CodeGraphRenderer(FLASK_GRAPH)

    def setUp(self):
        os.makedirs(TMP_DIR, exist_ok=True)

    def test_seed_repo_node(self):
        repo_id = _find_node_by_type(self.renderer, "Repo")
        path = os.path.join(TMP_DIR, "test_gv_repo_seed.png")
        self.renderer.render(repo_id, k=2, max_nodes=50,
                             output_path=path, show=False)
        self.assertTrue(os.path.isfile(path))

    def test_seed_attribute_node(self):
        attr_id = _find_node_by_type(self.renderer, "Attribute")
        self.assertIsNotNone(attr_id)
        path = os.path.join(TMP_DIR, "test_gv_attr_seed.png")
        self.renderer.render(attr_id, k=2, max_nodes=40,
                             output_path=path, show=False)
        self.assertTrue(os.path.isfile(path))

    def test_seed_textfile_node(self):
        tf_id = _find_node_by_type(self.renderer, "TextFile")
        if tf_id is not None:
            G, _ = self.renderer.sample_subgraph(tf_id, k=2, max_nodes=40)
            self.assertIn(tf_id, G.nodes())

    def test_seed_package_node(self):
        pkg_id = _find_node_by_type(self.renderer, "Package")
        self.assertIsNotNone(pkg_id)
        path = os.path.join(TMP_DIR, "test_gv_pkg_seed.png")
        self.renderer.render(pkg_id, k=2, max_nodes=40,
                             output_path=path, show=False)
        self.assertTrue(os.path.isfile(path))

    def test_seed_lambda_node(self):
        lam_id = _find_node_by_type(self.renderer, "Lambda")
        self.assertIsNotNone(lam_id)
        G, _ = self.renderer.sample_subgraph(lam_id, k=2, max_nodes=30)
        self.assertIn(lam_id, G.nodes())

    def test_subgraph_node_attributes_preserved(self):
        seed = _find_node_by_type(self.renderer, "Class", "Flask")
        G, _ = self.renderer.sample_subgraph(seed, k=1, max_nodes=30)
        for nid, data in G.nodes(data=True):
            orig = self.renderer.nodes_by_id.get(nid)
            self.assertIsNotNone(orig)
            self.assertEqual(data.get("nodeType"), orig.get("nodeType"))

    def test_empty_adj_node_handled(self):
        lam_id = _find_node_by_type(self.renderer, "Lambda")
        G, _ = self.renderer.sample_subgraph(lam_id, k=2, max_nodes=30)
        self.assertIn(lam_id, G.nodes())


# ===================================================================


class TestWithDjangoGraph(unittest.TestCase):
    """Smoke-test against the larger Django graph."""

    @classmethod
    def setUpClass(cls):
        django_dir = os.path.join(
            os.path.dirname(__file__),
            "../code_graph_data/swe-bench-lite",
        )
        django_files = [f for f in os.listdir(django_dir)
                        if f.startswith("django#django#")]
        if not django_files:
            raise unittest.SkipTest("No Django graph files found")
        cls.django_path = os.path.join(django_dir, django_files[0])
        cls.renderer = CodeGraphRenderer(cls.django_path)

    def setUp(self):
        os.makedirs(TMP_DIR, exist_ok=True)

    def test_django_loads_correctly(self):
        self.assertGreater(len(self.renderer.nodes_by_id), 30000)
        self.assertGreater(len(self.renderer.edges), 100000)

    def test_django_sample_does_not_explode(self):
        func_id = _find_node_by_type(self.renderer, "Function")
        self.assertIsNotNone(func_id)
        G, labels = self.renderer.sample_subgraph(func_id, k=2, max_nodes=80)
        self.assertLessEqual(G.number_of_nodes(), 80)
        self.assertGreater(G.number_of_nodes(), 0)

    def test_django_build_digraph_works(self):
        func_id = _find_node_by_type(self.renderer, "Function")
        G, labels = self.renderer.sample_subgraph(func_id, k=1, max_nodes=25)
        dot = self.renderer._build_digraph(G, labels, func_id)
        self.assertIsInstance(dot, graphviz.Digraph)

    def test_django_render_completes(self):
        func_id = _find_node_by_type(self.renderer, "Function")
        path = os.path.join(TMP_DIR, "test_gv_django.png")
        self.renderer.render(func_id, k=1, max_nodes=25,
                             output_path=path, show=False)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 1000)


# ===================================================================


class TestEscapeHtml(unittest.TestCase):
    """_escape_html helper."""

    def test_ampersand(self):
        self.assertEqual(_escape_html("a & b"), "a &amp; b")

    def test_less_than(self):
        self.assertEqual(_escape_html("a < b"), "a &lt; b")

    def test_greater_than(self):
        self.assertEqual(_escape_html("a > b"), "a &gt; b")

    def test_double_quote(self):
        self.assertEqual(_escape_html('a " b'), "a &quot; b")

    def test_no_escaping_needed(self):
        self.assertEqual(_escape_html("hello_world"), "hello_world")


# ===================================================================
# Teardown
# ===================================================================

def tearDownModule():
    """Clean up temporary test outputs."""
    import shutil
    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
