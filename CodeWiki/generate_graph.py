#!/usr/bin/env python3
"""
Code Repository Graph Generator

Parses a code repository via AST / tree-sitter static analysis and outputs a
graph JSON in the format of datatest.graph.json, with nodes (Repo, Package,
File, TextFile, Function, Class, Attribute, Lambda) and edges (contains,
calls, imports).

Supported languages: Python, JavaScript, TypeScript, Java, C, C++, C#, PHP, Kotlin.

Usage:
    python generate_graph.py <repo_path> [--output output.json] [--repo-name owner#repo#commit]
"""

import ast
import argparse
import fnmatch
import json
import logging
import os
import subprocess
import sys
import warnings
from collections import Counter
from typing import Optional, Union, Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exclude patterns
# ---------------------------------------------------------------------------

EXCLUDE_DIR_PATTERNS = {
    ".git", ".svn", ".hg",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis",
    ".tox", ".nox",
    "node_modules", ".npm", ".yarn", ".pnpm-store",
    ".next", ".nuxt", ".turbo",
    ".gradle",
    "venv", ".venv", "env", ".env", "virtualenv",
    ".idea", ".vscode", ".vs",
    ".cache", ".sass-cache",
    "bin", "obj",
    ".terraform",
    "build", "dist",
    ".eggs", "*.egg-info", "*.egg",
    ".circleci", ".github",
}

EXCLUDE_FILE_PATTERNS = {
    "*.pyc", "*.pyo", "*.pyd", "*.so",
    "*.class", "*.jar", "*.war", "*.ear",
    "*.o", "*.obj", "*.dll", "*.dylib", "*.exe", "*.lib", "*.a", "*.pdb",
    "*.min.js", "*.min.css", "*.map",
    "*.svg", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.pdf",
    "*.mov", "*.mp4", "*.mp3", "*.wav",
    "*.log", "*.bak", "*.swp", "*.tmp", "*.temp",
    "*.egg-info", "*.egg", "*.whl",
    "package-lock.json", "yarn.lock", "bun.lock", "bun.lockb",
    "poetry.lock", "Pipfile.lock",
    ".DS_Store", "Thumbs.db", "desktop.ini",
    "*.tfstate*",
}

PYTHON_BUILTINS = {
    "print", "len", "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "range", "enumerate", "zip", "isinstance", "hasattr", "getattr", "setattr",
    "open", "super", "__import__", "type", "object", "Exception", "ValueError",
    "TypeError", "KeyError", "IndexError", "AttributeError", "ImportError",
    "max", "min", "sum", "abs", "round", "sorted", "reversed", "filter", "map",
    "any", "all", "next", "iter", "callable", "repr", "format", "exec", "eval",
    "True", "False", "None",
}

# File extension → language name
CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c++": "cpp",
    ".hpp": "cpp", ".hxx": "cpp", ".hh": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".kt": "kotlin", ".kts": "kotlin",
}

# File extensions that should be parsed as code (not TextFile)
CODE_EXTENSIONS_SET = set(CODE_EXTENSIONS.keys())


def _match_pattern(name: str, patterns: set) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


# ---------------------------------------------------------------------------
# ID Allocator
# ---------------------------------------------------------------------------

class IDAllocator:
    def __init__(self, start: int = 10):
        self._next = start

    def allocate(self) -> int:
        idx = self._next
        self._next += 1
        return idx


# ---------------------------------------------------------------------------
# Graph Output Container
# ---------------------------------------------------------------------------

class GraphOutput:
    def __init__(self):
        self.nodes: list = []
        self.edges: list = []
        self._seen_edges: set = set()
        self.symbol_table: dict = {}  # name → [node_id, ...]
        self.pending_calls: list = []
        self.pending_imports: list = []
        self.file_paths: dict = {}  # file_id → absolute path

    def add_node(self, node_id: int, node_type: str, **fields) -> int:
        node = {"id": node_id, "nodeType": node_type}
        node.update(fields)
        self.nodes.append(node)
        return node_id

    def add_edge(self, edge_type: str, source: int, target: int):
        key = (edge_type, source, target)
        if key in self._seen_edges:
            return
        self._seen_edges.add(key)
        self.edges.append({"edgeType": edge_type, "source": source, "target": target})

    def add_symbol(self, name: str, node_id: int):
        self.symbol_table.setdefault(name, []).append(node_id)

    def to_json(self, output_path: str):
        result = {"edges": self.edges, "nodes": self.nodes}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
        logger.info(f"Graph written to {output_path}")
        logger.info(f"  Nodes: {len(self.nodes)}, Edges: {len(self.edges)}")


# ---------------------------------------------------------------------------
# Repository Walker
# ---------------------------------------------------------------------------

class RepoWalker:
    def __init__(self, repo_path: str, repo_name: str, id_alloc: IDAllocator, graph: GraphOutput,
                 exclude_extensions: Optional[set] = None):
        self.repo_path = repo_path
        self.repo_name = repo_name
        self.id_alloc = id_alloc
        self.graph = graph
        self.path_to_pkg_id: dict = {}
        self.exclude_extensions = exclude_extensions or set()

    def walk(self):
        repo_id = self.graph.add_node(
            self.id_alloc.allocate(), "Repo", repoName=self.repo_name, groupName=""
        )
        root_pkg_id = self.graph.add_node(self.id_alloc.allocate(), "Package", name="")
        self.graph.add_edge("contains", repo_id, root_pkg_id)
        self.path_to_pkg_id[""] = root_pkg_id

        for dirpath, dirnames, filenames in os.walk(self.repo_path, topdown=True):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and not _match_pattern(d, EXCLUDE_DIR_PATTERNS)
            ]

            rel = os.path.relpath(dirpath, self.repo_path)
            if rel == ".":
                rel = ""

            if rel != "":
                pkg_id = self.graph.add_node(self.id_alloc.allocate(), "Package", name=rel)
                parent_rel = os.path.dirname(rel)
                parent_id = self.path_to_pkg_id.get(parent_rel, root_pkg_id)
                self.graph.add_edge("contains", parent_id, pkg_id)
                self.path_to_pkg_id[rel] = pkg_id

            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                if _match_pattern(fname, EXCLUDE_FILE_PATTERNS):
                    continue

                abspath = os.path.join(dirpath, fname)
                if os.path.islink(abspath):
                    continue

                try:
                    with open(abspath, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except (OSError, PermissionError):
                    continue

                pkg_id = self.path_to_pkg_id.get(rel, root_pkg_id)

                # Determine file extension
                ext = os.path.splitext(fname)[1].lower()
                # Also handle compound extensions like .d.ts, .spec.ts
                if fname.endswith(".d.ts"):
                    ext = ".ts"
                elif fname.endswith(".spec.ts") or fname.endswith(".test.ts"):
                    ext = ".ts"

                # Skip user-excluded extensions (only checks simple extension)
                if ext in self.exclude_extensions:
                    continue
                # Also check compound extensions like ".spec.ts"
                if self.exclude_extensions and any(fname.endswith(e) for e in self.exclude_extensions if e.startswith(".") and "." in e[1:]):
                    continue

                if ext in CODE_EXTENSIONS_SET or fname.endswith(".py"):
                    # Re-check: use the right extension
                    if fname.endswith(".py"):
                        ext = ".py"
                    language = CODE_EXTENSIONS.get(ext, "unknown")
                    file_id = self.graph.add_node(
                        self.id_alloc.allocate(), "File",
                        fileName=fname, filePath=rel, text=text,
                    )
                    self.graph.file_paths[file_id] = (abspath, language)
                else:
                    file_id = self.graph.add_node(
                        self.id_alloc.allocate(), "TextFile",
                        name=fname, path=rel, text=text,
                    )

                self.graph.add_edge("contains", pkg_id, file_id)

        code_files = sum(1 for v in self.graph.file_paths.values() if v[1] != "unknown")
        logger.info(
            f"Repo walk complete: {len(self.path_to_pkg_id)} packages, "
            f"{code_files} code files, {len(self.graph.file_paths) - code_files} other files"
        )


# ---------------------------------------------------------------------------
# Tree-sitter Base Analyzer
# ---------------------------------------------------------------------------

class TreeSitterBase:
    """Shared utilities for tree-sitter language analyzers."""

    def __init__(self, file_id: int, file_path: str, content: str,
                 id_alloc: IDAllocator, graph: GraphOutput):
        self.file_id = file_id
        self.file_path = file_path
        self.content = content
        self.content_bytes = content.encode("utf-8")
        self.id_alloc = id_alloc
        self.graph = graph
        self.tree = None
        self.root = None

    def _init_parser(self, grammar_package, language_func: str = "language"):
        """Initialize tree-sitter parser with given grammar package."""
        from tree_sitter import Parser, Language
        try:
            capsule = getattr(grammar_package, language_func)()
            lang = Language(capsule)
            self.parser = Parser(lang)
        except Exception as e:
            logger.debug(f"Failed to init tree-sitter parser: {e}")
            self.parser = None

    def _parse(self) -> bool:
        if self.parser is None:
            return False
        try:
            self.tree = self.parser.parse(self.content_bytes)
            self.root = self.tree.root_node
            return True
        except Exception:
            return False

    def _node_text(self, node) -> str:
        try:
            return self.content_bytes[node.start_byte:node.end_byte].decode("utf-8")
        except Exception:
            return ""

    def _child_by_type(self, node, *type_names):
        for child in node.children:
            if child.type in type_names:
                return child
        return None

    def _children_by_type(self, node, *type_names):
        return [c for c in node.children if c.type in type_names]

    def _find_by_field(self, node, field_name: str):
        """Find child by field name (tree-sitter field)."""
        try:
            return node.child_by_field_name(field_name)
        except Exception:
            return None

    def _pos(self, node) -> tuple:
        """Return (start_line, start_col, end_line) 1-indexed."""
        sl = node.start_point[0] + 1
        sc = node.start_point[1]
        el = node.end_point[0] + 1
        return sl, sc, el

    def _comment(self, node) -> str:
        """Extract docstring-like comment from first statement of body."""
        body = self._child_by_type(node, "block", "statement_block", "class_body",
                                   "interface_body", "enum_body", "declaration_list",
                                   "compound_statement", "function_body", "body")
        if body is None:
            return "null"
        stmts = body.children
        if not stmts:
            return "null"
        first = stmts[0]
        if first.type in ("comment", "line_comment", "block_comment"):
            return self._node_text(first).strip()
        return "null"

    def _create_func(self, name: str, node, header_node=None, parent_id=None) -> int:
        sl, sc, el = self._pos(node)
        text = self._node_text(node)
        header_text = self._node_text(header_node) if header_node else text.split("\n")[0]
        func_id = self.id_alloc.allocate()
        self.graph.add_node(
            func_id, "Function",
            name=name, col=sc, startLoc=sl, endLoc=el,
            header=header_text, text=text, comment=self._comment(node),
        )
        pid = parent_id or self.file_id
        self.graph.add_edge("contains", pid, func_id)
        return func_id

    def _create_class(self, name: str, node, parent_id=None) -> int:
        sl, sc, el = self._pos(node)
        text = self._node_text(node)
        cls_id = self.id_alloc.allocate()
        self.graph.add_node(
            cls_id, "Class",
            className=name, col=sc, startLoc=sl, endLoc=el,
            text=text, comment=self._comment(node),
        )
        pid = parent_id or self.file_id
        self.graph.add_edge("contains", pid, cls_id)
        return cls_id

    def _create_attr(self, name: str, node, parent_id=None, attr_type="null") -> int:
        sl, sc, el = self._pos(node)
        text = self._node_text(node)
        attr_id = self.id_alloc.allocate()
        self.graph.add_node(
            attr_id, "Attribute",
            name=name, col=sc, startLoc=sl, endLoc=el,
            text=text, comment="null", attributeType=attr_type,
        )
        pid = parent_id or self.file_id
        self.graph.add_edge("contains", pid, attr_id)
        return attr_id

    def _create_lambda(self, text: str, node, parent_id=None) -> int:
        sl, sc, el = self._pos(node)
        lam_id = self.id_alloc.allocate()
        self.graph.add_node(
            lam_id, "Lambda",
            col=sc, startLoc=sl, endLoc=el, text=text,
        )
        pid = parent_id or self.file_id
        self.graph.add_edge("contains", pid, lam_id)
        return lam_id

    def _record_call(self, caller_id: int, callee_name: str, line: int):
        if callee_name:
            self.graph.pending_calls.append({
                "caller": caller_id, "callee_name": callee_name,
                "line": line, "file_id": self.file_id, "file_path": self.file_path,
            })

    def _record_import(self, name: str, module=None):
        if name:
            self.graph.pending_imports.append({
                "file_id": self.file_id, "module": module, "name": name,
                "asname": None, "line": 0,
            })

    def _walk(self, node, visitor, depth=0):
        """Generic recursive walk with a visitor callback. visitor(node, depth) → bool
        (return False to skip children)."""
        if visitor(node, depth):
            for child in node.children:
                self._walk(child, visitor, depth + 1)


# ===========================================================================
# Python File Analyzer (stdlib AST)
# ===========================================================================

class PythonFileAnalyzer(ast.NodeVisitor):
    def __init__(self, file_id: int, file_path: str, content: str,
                 id_alloc: IDAllocator, graph: GraphOutput):
        self.file_id = file_id
        self.file_path = file_path
        self.content = content
        self.lines = content.splitlines()
        self.id_alloc = id_alloc
        self.graph = graph
        self.current_class_id: Optional[int] = None
        self.current_function_id: Optional[int] = None
        self.scope_stack: List[Tuple[str, int]] = []

    def _extract_text(self, start_lineno: int, end_lineno: int) -> str:
        end = min(end_lineno, len(self.lines))
        if start_lineno < 1:
            start_lineno = 1
        return "\n".join(self.lines[start_lineno - 1 : end])

    def _extract_header(self, node: "Union[ast.FunctionDef, ast.AsyncFunctionDef]") -> str:
        if node.body:
            first_stmt = node.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                body_start = first_stmt.lineno
            else:
                body_start = first_stmt.lineno
        else:
            body_start = node.end_lineno + 1 if node.end_lineno else node.lineno + 1
        end = min(body_start - 1, len(self.lines))
        return "\n".join(self.lines[node.lineno - 1 : end])

    def visit_ClassDef(self, node: ast.ClassDef):
        docstring = ast.get_docstring(node)
        comment = docstring if docstring else "null"
        class_id = self.id_alloc.allocate()
        end_loc = node.end_lineno or node.lineno
        self.graph.add_node(
            class_id, "Class",
            className=node.name, col=node.col_offset,
            startLoc=node.lineno, endLoc=end_loc,
            text=self._extract_text(node.lineno, end_loc), comment=comment,
        )
        parent_id = self.current_class_id or self.current_function_id or self.file_id
        self.graph.add_edge("contains", parent_id, class_id)
        if not self.current_class_id and not self.current_function_id:
            self.graph.add_symbol(node.name, class_id)

        old_class = self.current_class_id
        self.current_class_id = class_id
        self.scope_stack.append(("class", class_id))
        self.generic_visit(node)
        self.scope_stack.pop()
        self.current_class_id = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)

    def _process_function(self, node: "Union[ast.FunctionDef, ast.AsyncFunctionDef]"):
        docstring = ast.get_docstring(node)
        comment = docstring if docstring else "null"
        header = self._extract_header(node)
        end_loc = node.end_lineno or node.lineno
        func_id = self.id_alloc.allocate()
        self.graph.add_node(
            func_id, "Function",
            name=node.name, col=node.col_offset,
            startLoc=node.lineno, endLoc=end_loc,
            header=header, text=self._extract_text(node.lineno, end_loc),
            comment=comment,
        )
        if self.current_class_id is not None:
            parent_id = self.current_class_id
        elif self.current_function_id is not None:
            parent_id = self.current_function_id
        else:
            parent_id = self.file_id
            self.graph.add_symbol(node.name, func_id)

        self.graph.add_edge("contains", parent_id, func_id)

        old_func = self.current_function_id
        self.current_function_id = func_id
        self.scope_stack.append(("function", func_id))
        self.generic_visit(node)
        self.scope_stack.pop()
        self.current_function_id = old_func

    def visit_Assign(self, node: ast.Assign):
        if self.current_function_id is not None:
            self.generic_visit(node)
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._create_attribute(target.id, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if self.current_function_id is not None:
            self.generic_visit(node)
            return
        if isinstance(node.target, ast.Name):
            self._create_attribute(node.target.id, node)
        self.generic_visit(node)

    def _create_attribute(self, name: str, node: ast.AST):
        attr_type = "null"
        if isinstance(node, ast.AnnAssign) and node.annotation:
            try:
                attr_type = ast.unparse(node.annotation)
            except Exception:
                pass
        elif isinstance(node, ast.Assign) and hasattr(node, 'value'):
            attr_type = self._infer_value_type(node.value)

        end_loc = node.end_lineno or node.lineno
        text = self._extract_text(node.lineno, end_loc) if hasattr(node, 'lineno') else ""
        attr_id = self.id_alloc.allocate()
        self.graph.add_node(
            attr_id, "Attribute",
            name=name, col=node.col_offset,
            startLoc=node.lineno, endLoc=end_loc,
            text=text, comment="null", attributeType=attr_type,
        )
        parent_id = self.current_class_id or self.file_id
        self.graph.add_edge("contains", parent_id, attr_id)
        if not self.current_class_id:
            self.graph.add_symbol(name, attr_id)

    def _infer_value_type(self, value: ast.AST) -> str:
        if isinstance(value, ast.Constant):
            tn = type(value.value).__name__
            return "None" if tn == "NoneType" else tn
        if isinstance(value, (ast.List, ast.ListComp)):
            return "list"
        if isinstance(value, (ast.Dict, ast.DictComp)):
            return "dict"
        if isinstance(value, (ast.Set, ast.SetComp)):
            return "set"
        if isinstance(value, ast.Tuple):
            return "tuple"
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                return value.func.id
            if isinstance(value.func, ast.Attribute):
                return value.func.attr
        if isinstance(value, ast.Lambda):
            return "lambda"
        if isinstance(value, ast.Name):
            return value.id
        return "null"

    def visit_Lambda(self, node: ast.Lambda):
        lam_id = self.id_alloc.allocate()
        end_loc = node.end_lineno or node.lineno
        text = self._extract_text(node.lineno, end_loc) if hasattr(node, 'lineno') else ""
        if not text or text.strip() == "":
            try:
                text = ast.unparse(node)
            except Exception:
                text = "lambda ..."
        self.graph.add_node(
            lam_id, "Lambda",
            col=node.col_offset,
            startLoc=node.lineno if hasattr(node, 'lineno') else 0,
            endLoc=end_loc, text=text,
        )
        parent_id = self.current_function_id or self.current_class_id or self.file_id
        self.graph.add_edge("contains", parent_id, lam_id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        call_name = self._get_call_name(node.func)
        if call_name:
            caller_id = self._find_caller_id()
            if caller_id is not None:
                self.graph.pending_calls.append({
                    "caller": caller_id, "callee_name": call_name,
                    "line": node.lineno, "file_id": self.file_id,
                    "file_path": self.file_path,
                })
        self.generic_visit(node)

    def _find_caller_id(self) -> Optional[int]:
        for kind, sid in reversed(self.scope_stack):
            if kind in ("function", "lambda"):
                return sid
        return self.current_function_id or self.current_class_id

    def _get_call_name(self, node) -> Optional[str]:
        if isinstance(node, ast.Name):
            if node.id in PYTHON_BUILTINS:
                return None
            return node.id
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in PYTHON_BUILTINS:
                    return None
                return f"{node.value.id}.{node.attr}"
            if isinstance(node.value, ast.Attribute):
                base = self._get_call_name(node.value)
                if base:
                    return f"{base}.{node.attr}"
            return node.attr
        if isinstance(node, ast.Subscript):
            return self._get_call_name(node.value)
        return None

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.graph.pending_imports.append({
                "file_id": self.file_id, "module": None,
                "name": alias.name, "asname": alias.asname, "line": node.lineno,
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level is not None and node.level > 0:
            self.generic_visit(node)
            return
        module = node.module or ""
        for alias in node.names:
            self.graph.pending_imports.append({
                "file_id": self.file_id, "module": module,
                "name": alias.name, "asname": alias.asname, "line": node.lineno,
            })
        self.generic_visit(node)

    def analyze(self):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                tree = ast.parse(self.content)
            self.visit(tree)
        except SyntaxError:
            logger.debug(f"Syntax error in {self.file_path}, skipping")
        except Exception:
            logger.debug(f"Error analyzing {self.file_path}", exc_info=True)


# ===========================================================================
# Tree-sitter: JavaScript / TypeScript Analyzer
# ===========================================================================

class _JSBaseAnalyzer(TreeSitterBase):
    """Shared logic for JavaScript and TypeScript."""

    def _extract_all(self):
        if not self._parse():
            return
        # Pass 1: collect entities
        self._collect_entities(self.root)
        # Pass 2: collect relationships
        self._collect_relations(self.root, None)

    def _collect_entities(self, node, parent_id=None):
        """Recursively collect Function, Class, Attribute from the tree."""
        pid = parent_id or self.file_id
        t = node.type

        if t == "class_declaration" or t == "abstract_class_declaration":
            name_node = self._child_by_type(node, "identifier", "type_identifier")
            if name_node:
                cls_id = self._create_class(self._node_text(name_node), node, pid)
                body = self._child_by_type(node, "class_body", "declaration_list")
                if body:
                    self._collect_entities(body, cls_id)
                return

        if t == "function_declaration" or t == "generator_function_declaration":
            name_node = self._child_by_type(node, "identifier")
            if name_node:
                func_name = self._node_text(name_node)
                func_id = self._create_func(func_name, node, parent_id=pid)
                if parent_id == self.file_id:
                    self.graph.add_symbol(func_name, func_id)
                # Recurse into body for nested functions
                body = self._child_by_type(node, "statement_block")
                if body:
                    self._collect_entities(body, func_id)
                return

        if t == "method_definition":
            name_node = self._child_by_type(node, "property_identifier")
            if not name_node:
                name_node = self._child_by_type(node, "identifier")
            if name_node:
                self._create_func(self._node_text(name_node), node, parent_id=pid)
                body = self._child_by_type(node, "statement_block")
                if body:
                    self._collect_entities(body, parent_id)
                return

        if t in ("lexical_declaration", "variable_declaration",
                 "const_declaration", "let_declaration", "var_declaration"):
            if parent_id is not None:
                for decl in self._children_by_type(node, "variable_declarator"):
                    name_node = self._child_by_type(decl, "identifier")
                    if name_node:
                        val = self._child_by_type(decl, "arrow_function")
                        if val:
                            func_id = self._create_func(self._node_text(name_node), node, parent_id=pid)
                            if parent_id == self.file_id:
                                self.graph.add_symbol(self._node_text(name_node), func_id)
                            body = self._child_by_type(val, "statement_block")
                            if body:
                                self._collect_entities(body, func_id)
                        else:
                            self._create_attr(self._node_text(name_node), decl, parent_id=pid)
                            if parent_id == self.file_id:
                                self.graph.add_symbol(self._node_text(name_node), node)
                return

        if t == "export_statement":
            # Delegate to inner declaration
            inner = self._child_by_type(node, "function_declaration", "class_declaration",
                                        "lexical_declaration", "variable_declaration")
            if inner:
                self._collect_entities(inner, parent_id)
            return

        # Arrow function / function expression assigned to variable → Lambda
        if t == "arrow_function":
            self._create_lambda(self._node_text(node), node, parent_id=pid)
            body = self._child_by_type(node, "statement_block")
            if body:
                self._collect_entities(body, parent_id)
            return

        # Recursively walk children for nested structures
        for child in node.children:
            self._collect_entities(child, parent_id)

    def _collect_relations(self, node, current_func_id):
        t = node.type

        if t in ("function_declaration", "generator_function_declaration",
                 "method_definition", "arrow_function"):
            current_func_id = self._find_enclosing(node)
            if current_func_id is None:
                # Find the function node via name
                name_node = self._child_by_type(node, "identifier", "property_identifier")
                if name_node:
                    current_func_id = node.id if hasattr(node, 'id') else None

        if t == "call_expression":
            callee = self._extract_js_callee(node)
            if callee and current_func_id:
                sl = node.start_point[0] + 1
                self._record_call(current_func_id, callee, sl)

        if t in ("import_statement", "import_declaration"):
            import_clause = self._child_by_type(node, "import_clause")
            if import_clause:
                for id_node in self._children_by_type(import_clause, "identifier"):
                    self._record_import(self._node_text(id_node))

        for child in node.children:
            self._collect_relations(child, current_func_id)

    def _extract_js_callee(self, call_node) -> Optional[str]:
        func_node = self._child_by_type(call_node, "identifier",
                                        "member_expression", "property_identifier")
        if func_node is None:
            func_node = call_node.children[0] if call_node.children else None
        if func_node is None:
            return None
        if func_node.type == "identifier":
            return self._node_text(func_node)
        if func_node.type == "member_expression":
            obj = self._child_by_type(func_node, "identifier")
            prop = self._child_by_type(func_node, "property_identifier")
            if obj and prop:
                return f"{self._node_text(obj)}.{self._node_text(prop)}"
            if prop:
                return self._node_text(prop)
            return self._node_text(func_node).split(".")[-1] if "." in self._node_text(func_node) else self._node_text(func_node)
        if func_node.type == "property_identifier":
            return self._node_text(func_node)
        return None

    def _find_enclosing(self, node):
        p = node.parent
        while p:
            if p.type in ("function_declaration", "method_definition", "arrow_function"):
                name_node = self._child_by_type(p, "identifier", "property_identifier")
                if name_node:
                    return name_node.id  # won't match global symbol table
                return p.id
            p = p.parent
        return None


class JavaScriptAnalyzer(_JSBaseAnalyzer):
    def analyze(self):
        try:
            import tree_sitter_javascript
            self._init_parser(tree_sitter_javascript)
            self._extract_all()
        except ImportError:
            logger.debug("tree_sitter_javascript not available")
        except Exception:
            logger.debug(f"Error analyzing JS {self.file_path}", exc_info=True)


class TypeScriptAnalyzer(_JSBaseAnalyzer):
    def analyze(self):
        try:
            import tree_sitter_typescript
            self._init_parser(tree_sitter_typescript, "language_typescript")
            self._extract_all()
        except ImportError:
            logger.debug("tree_sitter_typescript not available")
        except Exception:
            logger.debug(f"Error analyzing TS {self.file_path}", exc_info=True)


# ===========================================================================
# Tree-sitter: Java Analyzer
# ===========================================================================

class JavaAnalyzer(TreeSitterBase):
    def analyze(self):
        try:
            import tree_sitter_java
            self._init_parser(tree_sitter_java)
        except ImportError:
            logger.debug("tree_sitter_java not available")
            return
        if not self._parse():
            return
        self._extract_entities(self.root, self.file_id)
        self._extract_relations(self.root, self.file_id)

    def _extract_entities(self, node, parent_id):
        t = node.type

        if t == "class_declaration":
            name_node = self._child_by_type(node, "identifier")
            if name_node:
                cls_id = self._create_class(self._node_text(name_node), node, parent_id)
                if parent_id == self.file_id:
                    self.graph.add_symbol(self._node_text(name_node), cls_id)
                body = self._child_by_type(node, "class_body")
                if body:
                    self._extract_entities(body, cls_id)
                return

        if t == "interface_declaration":
            name_node = self._child_by_type(node, "identifier")
            if name_node:
                cls_id = self._create_class(self._node_text(name_node), node, parent_id)
                if parent_id == self.file_id:
                    self.graph.add_symbol(self._node_text(name_node), cls_id)
                body = self._child_by_type(node, "interface_body")
                if body:
                    self._extract_entities(body, cls_id)
                return

        if t == "method_declaration":
            name_node = self._child_by_type(node, "identifier")
            if name_node:
                self._create_func(self._node_text(name_node), node, parent_id=parent_id)
            return

        if t == "field_declaration":
            for decl in self._children_by_type(node, "variable_declarator"):
                name_node = self._child_by_type(decl, "identifier")
                type_node = self._child_by_type(node, "type_identifier")
                if name_node:
                    atype = self._node_text(type_node) if type_node else "null"
                    attr_id = self._create_attr(self._node_text(name_node), decl, parent_id, attr_type=atype)
                    if parent_id == self.file_id:
                        self.graph.add_symbol(self._node_text(name_node), attr_id)
            return

        for child in node.children:
            self._extract_entities(child, parent_id)

    def _extract_relations(self, node, current_func_id):
        t = node.type

        if t == "method_declaration":
            name_node = self._child_by_type(node, "identifier")
            if name_node:
                current_func_id = name_node.id if hasattr(name_node, 'id') else current_func_id

        if t == "method_invocation":
            callee_node = self._child_by_type(node, "identifier")
            if callee_node:
                callee = self._node_text(callee_node)
                if current_func_id and callee:
                    sl = node.start_point[0] + 1
                    self._record_call(current_func_id, callee, sl)

        if t == "import_declaration":
            # import com.foo.Bar → extract "Bar"
            for id_node in self._children_by_type(node, "identifier"):
                self._record_import(self._node_text(id_node))

        for child in node.children:
            self._extract_relations(child, current_func_id)


# ===========================================================================
# Tree-sitter: C / C++ Analyzer
# ===========================================================================

class CAnalyzer(TreeSitterBase):
    def analyze(self):
        try:
            import tree_sitter_c
            self._init_parser(tree_sitter_c)
        except ImportError:
            logger.debug("tree_sitter_c not available")
            return
        if not self._parse():
            return
        self._extract_entities(self.root, self.file_id)
        self._extract_relations(self.root, self.file_id)

    def _extract_entities(self, node, parent_id):
        t = node.type

        if t == "function_definition":
            declarator = self._child_by_type(node, "function_declarator")
            if declarator:
                id_node = self._child_by_type(declarator, "identifier")
                # nested pointer_declarator → function_declarator → identifier
                if not id_node:
                    ptr = self._child_by_type(declarator, "pointer_declarator")
                    if ptr:
                        fd = self._child_by_type(ptr, "function_declarator")
                        if fd:
                            id_node = self._child_by_type(fd, "identifier")
                if id_node:
                    func_id = self._create_func(self._node_text(id_node), node, parent_id=parent_id)
                    if parent_id == self.file_id:
                        self.graph.add_symbol(self._node_text(id_node), func_id)
                    body = self._child_by_type(node, "compound_statement")
                    if body:
                        self._extract_entities(body, func_id)
            return

        if t == "preproc_function_def":
            id_node = self._child_by_type(node, "identifier")
            if id_node:
                self._create_func(self._node_text(id_node), node, parent_id=parent_id)
            return

        for child in node.children:
            self._extract_entities(child, parent_id)

    def _extract_relations(self, node, current_func_id):
        t = node.type

        if t == "function_definition":
            declarator = self._child_by_type(node, "function_declarator")
            if declarator:
                id_node = self._child_by_type(declarator, "identifier")
                if not id_node:
                    ptr = self._child_by_type(declarator, "pointer_declarator")
                    if ptr:
                        fd = self._child_by_type(ptr, "function_declarator")
                        if fd:
                            id_node = self._child_by_type(fd, "identifier")
                if id_node:
                    current_func_id = id_node.id if hasattr(id_node, 'id') else current_func_id

        if t == "call_expression":
            func_node = self._child_by_type(node, "identifier")
            if not func_node:
                mem = self._child_by_type(node, "field_expression")
                if mem:
                    func_node = self._child_by_type(mem, "field_identifier")
            if func_node and current_func_id:
                callee = self._node_text(func_node)
                sl = node.start_point[0] + 1
                self._record_call(current_func_id, callee, sl)

        for child in node.children:
            self._extract_relations(child, current_func_id)


class CppAnalyzer(CAnalyzer):
    def analyze(self):
        try:
            import tree_sitter_cpp
            self._init_parser(tree_sitter_cpp)
        except ImportError:
            logger.debug("tree_sitter_cpp not available")
            return
        if not self._parse():
            return
        self._extract_entities(self.root, self.file_id)
        self._extract_relations(self.root, self.file_id)

    def _extract_entities(self, node, parent_id):
        t = node.type

        if t == "class_specifier":
            name_node = self._child_by_type(node, "type_identifier")
            if name_node:
                cls_id = self._create_class(self._node_text(name_node), node, parent_id)
                if parent_id == self.file_id:
                    self.graph.add_symbol(self._node_text(name_node), cls_id)
                body = self._child_by_type(node, "field_declaration_list")
                if body:
                    self._extract_entities(body, cls_id)
                return

        if t == "struct_specifier":
            name_node = self._child_by_type(node, "type_identifier")
            if name_node:
                cls_id = self._create_class(self._node_text(name_node), node, parent_id)
                if parent_id == self.file_id:
                    self.graph.add_symbol(self._node_text(name_node), cls_id)
                body = self._child_by_type(node, "field_declaration_list")
                if body:
                    self._extract_entities(body, cls_id)
                return

        if t == "field_declaration":
            # Member variable
            for decl in self._children_by_type(node, "field_identifier"):
                self._create_attr(self._node_text(decl), decl, parent_id)
            for decl in self._children_by_type(node, "identifier"):
                # Could be a more complex declarator
                pass
            return

        # Fall through to C base class for function_definition
        super()._extract_entities(node, parent_id)

    def _extract_relations(self, node, current_func_id):
        t = node.type

        if t == "class_specifier" or t == "struct_specifier":
            name_node = self._child_by_type(node, "type_identifier")
            if name_node:
                current_func_id = name_node.id if hasattr(name_node, 'id') else current_func_id

        super()._extract_relations(node, current_func_id)


# ===========================================================================
# Tree-sitter: C# Analyzer
# ===========================================================================

class CSharpAnalyzer(TreeSitterBase):
    def analyze(self):
        try:
            import tree_sitter_c_sharp
            self._init_parser(tree_sitter_c_sharp)
        except ImportError:
            logger.debug("tree_sitter_c_sharp not available")
            return
        if not self._parse():
            return
        self._extract_all(self.root, self.file_id, None)

    def _extract_all(self, node, parent_id, current_func_id):
        t = node.type
        pid = parent_id
        cid = current_func_id

        if t == "class_declaration":
            name_node = self._child_by_type(node, "identifier")
            if name_node:
                cls_id = self._create_class(self._node_text(name_node), node, pid)
                if pid == self.file_id:
                    self.graph.add_symbol(self._node_text(name_node), cls_id)
                pid = cls_id

        elif t == "method_declaration":
            name_node = self._child_by_type(node, "identifier")
            if name_node:
                func_id = self._create_func(self._node_text(name_node), node, parent_id=pid)
                cid = func_id

        elif t == "field_declaration":
            for decl in self._children_by_type(node, "variable_declarator"):
                name_node = self._child_by_type(decl, "identifier")
                if name_node:
                    type_node = self._child_by_type(node, "identifier", "predefined_type")
                    atype = self._node_text(type_node) if type_node else "null"
                    self._create_attr(self._node_text(name_node), decl, pid, attr_type=atype)
                    if pid == self.file_id:
                        self.graph.add_symbol(self._node_text(name_node), node)

        elif t == "property_declaration":
            name_node = self._child_by_type(node, "identifier")
            if name_node:
                self._create_attr(self._node_text(name_node), node, pid)
                if pid == self.file_id:
                    self.graph.add_symbol(self._node_text(name_node), node)

        elif t == "invocation_expression":
            func_node = self._child_by_type(node, "identifier", "member_access_expression")
            if func_node and cid:
                if func_node.type == "identifier":
                    callee = self._node_text(func_node)
                else:
                    prop = self._child_by_type(func_node, "identifier")
                    callee = self._node_text(prop) if prop else ""
                if callee:
                    sl = node.start_point[0] + 1
                    self._record_call(cid, callee, sl)

        elif t == "using_directive":
            for id_node in self._children_by_type(node, "identifier"):
                self._record_import(self._node_text(id_node))

        for child in node.children:
            self._extract_all(child, pid, cid)


# ===========================================================================
# Tree-sitter: PHP Analyzer
# ===========================================================================

class PhpAnalyzer(TreeSitterBase):
    def analyze(self):
        try:
            import tree_sitter_php
            self._init_parser(tree_sitter_php, "language_php")
        except ImportError:
            logger.debug("tree_sitter_php not available")
            return
        if not self._parse():
            return
        self._extract_all(self.root, self.file_id, None)

    def _extract_all(self, node, parent_id, current_func_id):
        t = node.type
        pid = parent_id
        cid = current_func_id

        if t == "class_declaration":
            name_node = self._child_by_type(node, "name")
            if not name_node:
                name_node = self._child_by_type(node, "identifier")
            if name_node:
                cls_id = self._create_class(self._node_text(name_node), node, pid)
                if pid == self.file_id:
                    self.graph.add_symbol(self._node_text(name_node), cls_id)
                pid = cls_id

        elif t == "function_definition":
            name_node = self._child_by_type(node, "name")
            if not name_node:
                name_node = self._child_by_type(node, "identifier")
            if name_node:
                func_name = self._node_text(name_node)
                func_id = self._create_func(func_name, node, parent_id=pid)
                if pid == self.file_id:
                    self.graph.add_symbol(func_name, func_id)
                cid = func_id

        elif t == "method_declaration":
            name_node = self._child_by_type(node, "name")
            if not name_node:
                name_node = self._child_by_type(node, "identifier")
            if name_node:
                self._create_func(self._node_text(name_node), node, parent_id=pid)

        elif t == "function_call_expression":
            func_node = self._child_by_type(node, "name")
            if not func_node:
                func_node = self._child_by_type(node, "identifier")
            if func_node and cid:
                callee = self._node_text(func_node)
                sl = node.start_point[0] + 1
                self._record_call(cid, callee, sl)

        elif t == "use_declaration":
            for id_node in self._children_by_type(node, "name", "identifier"):
                self._record_import(self._node_text(id_node))

        elif t == "include_expression":
            pass  # File-level include, skip

        for child in node.children:
            self._extract_all(child, pid, cid)


# ===========================================================================
# Tree-sitter: Kotlin Analyzer
# ===========================================================================

class KotlinAnalyzer(TreeSitterBase):
    def analyze(self):
        try:
            import tree_sitter_kotlin
            self._init_parser(tree_sitter_kotlin)
        except ImportError:
            logger.debug("tree_sitter_kotlin not available")
            return
        if not self._parse():
            return
        self._extract_all(self.root, self.file_id, None)

    def _extract_all(self, node, parent_id, current_func_id):
        t = node.type
        pid = parent_id
        cid = current_func_id

        if t == "class_declaration":
            name_node = self._child_by_type(node, "type_identifier", "identifier")
            if name_node:
                cls_id = self._create_class(self._node_text(name_node), node, pid)
                if pid == self.file_id:
                    self.graph.add_symbol(self._node_text(name_node), cls_id)
                pid = cls_id

        elif t == "function_declaration":
            name_node = self._child_by_type(node, "simple_identifier")
            if not name_node:
                name_node = self._child_by_type(node, "identifier")
            if name_node:
                func_name = self._node_text(name_node)
                func_id = self._create_func(func_name, node, parent_id=pid)
                if pid == self.file_id:
                    self.graph.add_symbol(func_name, func_id)
                cid = func_id

        elif t == "property_declaration":
            name_node = self._child_by_type(node, "simple_identifier", "identifier")
            if name_node:
                type_node = self._child_by_type(node, "type_identifier", "user_type")
                atype = self._node_text(type_node) if type_node else "null"
                self._create_attr(self._node_text(name_node), node, pid, attr_type=atype)
                if pid == self.file_id:
                    self.graph.add_symbol(self._node_text(name_node), node)

        elif t == "call_expression":
            func_node = self._child_by_type(node, "simple_identifier",
                                            "identifier", "navigation_expression")
            if func_node and cid:
                if func_node.type == "navigation_expression":
                    prop = self._child_by_type(func_node, "simple_identifier", "identifier")
                    callee = self._node_text(prop) if prop else self._node_text(func_node).split(".")[-1]
                else:
                    callee = self._node_text(func_node)
                sl = node.start_point[0] + 1
                self._record_call(cid, callee, sl)

        elif t == "lambda_literal":
            self._create_lambda(self._node_text(node), node, pid)

        elif t == "import_header":
            for id_node in self._children_by_type(node, "identifier"):
                self._record_import(self._node_text(id_node))

        for child in node.children:
            self._extract_all(child, pid, cid)


# ===========================================================================
# Analyzer Registry
# ===========================================================================

LANGUAGE_ANALYZERS = {
    "python": PythonFileAnalyzer,
    "javascript": JavaScriptAnalyzer,
    "typescript": TypeScriptAnalyzer,
    "java": JavaAnalyzer,
    "c": CAnalyzer,
    "cpp": CppAnalyzer,
    "csharp": CSharpAnalyzer,
    "php": PhpAnalyzer,
    "kotlin": KotlinAnalyzer,
}


def analyze_file(file_id: int, file_path: str, content: str, language: str,
                 id_alloc: IDAllocator, graph: GraphOutput):
    """Dispatch to the correct language analyzer."""
    cls = LANGUAGE_ANALYZERS.get(language)
    if cls is None:
        logger.debug(f"No analyzer for language: {language} ({file_path})")
        return

    try:
        analyzer = cls(file_id, file_path, content, id_alloc, graph)
        analyzer.analyze()
    except Exception:
        logger.debug(f"Error analyzing {file_path}", exc_info=True)


# ---------------------------------------------------------------------------
# Cross-file Resolution
# ---------------------------------------------------------------------------

def resolve_calls(graph: GraphOutput):
    resolved = 0
    for call in graph.pending_calls:
        callee = call["callee_name"]
        targets = []
        if callee in graph.symbol_table:
            targets = graph.symbol_table[callee]
        elif "." in callee:
            parts = callee.split(".")
            if parts[-1] in graph.symbol_table:
                targets = graph.symbol_table[parts[-1]]
        for tid in targets:
            if tid != call["caller"]:
                graph.add_edge("calls", call["caller"], tid)
                resolved += 1
    logger.info(f"Resolved {resolved} call edges from {len(graph.pending_calls)} pending calls")


def resolve_imports(graph: GraphOutput):
    resolved = 0
    for imp in graph.pending_imports:
        name = imp["name"]
        if name in graph.symbol_table:
            for tid in graph.symbol_table[name]:
                graph.add_edge("imports", imp["file_id"], tid)
                resolved += 1
    logger.info(f"Resolved {resolved} import edges from {len(graph.pending_imports)} pending imports")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def detect_repo_name(repo_path: str) -> str:
    owner = "unknown"
    repo = os.path.basename(os.path.abspath(repo_path))
    commit = "unknown"

    try:
        remote_url = subprocess.check_output(
            ["git", "-C", repo_path, "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        remote_url = remote_url.rstrip("/")
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]
        if ":" in remote_url and "://" not in remote_url:
            path_part = remote_url.split(":")[-1]
        else:
            path_part = remote_url.split("://")[-1].split("/", 1)[-1] if "://" in remote_url else remote_url
        parts = path_part.strip("/").split("/")
        if len(parts) >= 2:
            owner = parts[-2]
            repo = parts[-1]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        pass

    try:
        commit = subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return f"{owner}#{repo}#{commit}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a code dependency graph JSON from a repository "
                    "(Python, JavaScript, TypeScript, Java, C, C++, C#, PHP, Kotlin)"
    )
    parser.add_argument("repo_path", help="Path to the repository to analyze")
    parser.add_argument("--output", "-o", default="output.graph.json", help="Output JSON file path")
    parser.add_argument("--repo-name", help="Repo name in owner#repo#commit format (auto-detected if omitted)")
    parser.add_argument("--exclude-ext", nargs="*", default=None,
                        help="File extensions to skip (e.g. --exclude-ext .spec.ts .test.py .d.ts)")
    args = parser.parse_args()

    # Normalize exclude extensions to lowercase with leading dot
    exclude_extensions = set()
    if args.exclude_ext:
        for e in args.exclude_ext:
            e = e.strip().lower()
            if not e.startswith("."):
                e = "." + e
            exclude_extensions.add(e)
        logger.info(f"Excluding extensions: {sorted(exclude_extensions)}")

    repo_path = os.path.abspath(args.repo_path)
    if not os.path.isdir(repo_path):
        logger.error(f"Repository path not found: {repo_path}")
        sys.exit(1)

    repo_name = args.repo_name or detect_repo_name(repo_path)
    logger.info(f"Analyzing repository: {repo_name}")
    logger.info(f"Repo path: {repo_path}")

    id_alloc = IDAllocator(start=10)
    graph = GraphOutput()

    # Phase 1: Walk repository
    logger.info("Phase 1: Walking repository tree...")
    walker = RepoWalker(repo_path, repo_name, id_alloc, graph, exclude_extensions)
    walker.walk()

    # Phase 2: Parse code files with language-specific analyzers
    files_by_lang = {}
    for file_id, (abs_path, language) in graph.file_paths.items():
        files_by_lang.setdefault(language, []).append((file_id, abs_path))

    logger.info(f"Phase 2: Parsing {len(graph.file_paths)} code files in {len(files_by_lang)} languages...")
    for lang, files in sorted(files_by_lang.items()):
        logger.info(f"  Language '{lang}': {len(files)} files")

    total_parsed = 0
    total_failed = 0
    for lang, files in sorted(files_by_lang.items()):
        for idx, (file_id, abs_path) in enumerate(files, 1):
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except (OSError, PermissionError):
                total_failed += 1
                continue

            analyze_file(file_id, abs_path, content, lang, id_alloc, graph)
            total_parsed += 1

            if total_parsed % 100 == 0:
                logger.info(f"  Parsed {total_parsed}/{len(graph.file_paths)} files...")

    logger.info(f"  Parsed {total_parsed} files ({total_failed} failed)")

    # Phase 3: Cross-file resolution
    logger.info("Phase 3: Resolving cross-file references...")
    logger.info(f"  Symbol table: {len(graph.symbol_table)} unique names, "
                f"{sum(len(v) for v in graph.symbol_table.values())} total entries")
    resolve_calls(graph)
    resolve_imports(graph)

    # Phase 4: Write output
    logger.info("Phase 4: Writing output...")
    graph.to_json(args.output)

    # Summary
    node_types = Counter(n["nodeType"] for n in graph.nodes)
    edge_types = Counter(e["edgeType"] for e in graph.edges)
    logger.info("Node type counts:")
    for nt, count in sorted(node_types.items()):
        logger.info(f"  {nt}: {count}")
    logger.info("Edge type counts:")
    for et, count in sorted(edge_types.items()):
        logger.info(f"  {et}: {count}")


if __name__ == "__main__":
    main()
