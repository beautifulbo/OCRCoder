#!/usr/bin/env python3
"""
Generate module_tree.json for a given repository.
Usage:
    python generate_module_tree.py /path/to/repo [--output ./output] [--main-model MODEL] [--cluster-model MODEL]
"""

import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate module_tree.json for a code repository using CodeWiki's analysis + LLM-powered clustering."
    )
    parser.add_argument("--repo_path", type=str, help="Path to the repository to analyze")
    parser.add_argument("--output", type=str, default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--main-model", type=str, default="deepseek-v4-flash", help="Main LLM model (default: claude-sonnet-4)")
    parser.add_argument("--cluster-model", type=str, default=None, help="Clustering LLM model (default: same as --main-model)")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="LLM API base URL (default: http://0.0.0.0:4000/)")
    parser.add_argument("--api-key", type=str, default="sk-e92669a424aa40d8b478dbefc2e95e4a", help="LLM API key (default: sk-1234)")
    parser.add_argument("--provider", type=str, default="openai-compatible",
                        choices=["openai-compatible", "anthropic", "bedrock", "azure-openai", "claude-code", "codex"],
                        help="LLM provider (default: openai-compatible)")

    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    if not os.path.isdir(repo_path):
        logger.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output)
    cluster_model = args.cluster_model or args.main_model

    logger.info(f"📂 Repository: {repo_path}")
    logger.info(f"📁 Output:     {output_dir}")
    logger.info(f"🤖 Main model: {args.main_model}")
    logger.info(f"🔀 Cluster:    {cluster_model}")
    logger.info(f"🔌 Provider:   {args.provider}")
    logger.info("")

    # ------------------------------------------------------------
    # Step 1: Build backend Config
    # ------------------------------------------------------------
    from codewiki.src.config import Config, set_cli_context
    set_cli_context(True)

    config = Config.from_cli(
        repo_path=repo_path,
        output_dir=output_dir,
        llm_base_url=args.base_url,
        llm_api_key=args.api_key,
        main_model=args.main_model,
        cluster_model=cluster_model,
        provider=args.provider,
    )

    # ------------------------------------------------------------
    # Step 2: Build dependency graph (pure code analysis, no LLM)
    # ------------------------------------------------------------
    logger.info("🔍 Phase 1: Parsing repository and building dependency graph...")
    from codewiki.src.be.dependency_analyzer import DependencyGraphBuilder

    graph_builder = DependencyGraphBuilder(config)
    components, leaf_nodes = graph_builder.build_dependency_graph()

    logger.info(f"   ✅ Found {len(components)} components, {len(leaf_nodes)} leaf nodes")
    logger.info("")

    # ------------------------------------------------------------
    # Step 3: Get LLM backend and cluster modules
    # ------------------------------------------------------------
    logger.info("🧠 Phase 2: Clustering modules via LLM...")
    from codewiki.src.be.backend import get_backend
    from codewiki.src.be.cluster_modules import cluster_modules
    from codewiki.src.utils import file_manager

    backend = get_backend(config)

    module_tree = cluster_modules(
        leaf_nodes,
        components,
        config,
        completer=lambda p: backend.complete(p, model=cluster_model),
    )

    # ------------------------------------------------------------
    # Step 4: Save module_tree.json
    # ------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    module_tree_path = os.path.join(output_dir, "module_tree.json")
    file_manager.save_json(module_tree, module_tree_path)

    logger.info(f"   ✅ module_tree.json saved to {module_tree_path}")
    logger.info(f"   📊 Total top-level modules: {len(module_tree)}")

    # Print a quick summary
    def print_tree(tree, indent=0):
        for name, info in tree.items():
            comp_count = len(info.get("components", []))
            children = info.get("children", {})
            child_count = len(children) if isinstance(children, dict) else 0
            logger.info(f"   {'  ' * indent}- {name} ({comp_count} components, {child_count} children)")
            if isinstance(children, dict) and children:
                print_tree(children, indent + 1)

    logger.info("")
    logger.info("📋 Module tree summary:")
    print_tree(module_tree)


if __name__ == "__main__":
    main()
