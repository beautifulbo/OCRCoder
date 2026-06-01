"""
CGM Benchmark 纯文本评测引擎 — 使用 Qwen3-8B 进行代码补全。

与 evaluate.py (Qwen3-VL, 图像输入) 形成对比:
  - 将代码子图序列化为结构化文本，代替 Graphviz 渲染的 PNG 图像
  - 使用纯文本 LLM (Qwen3ForCausalLM) 代替视觉语言模型 (Qwen3VLForConditionalGeneration)
  - 移除所有图像处理依赖 (Graphviz, qwen-vl-utils, process_vision_info)

用法:
  # Mock 模式（测试评测管线）
  python evaluate_image.py --mock --limit 20

  # 纯文本模型评测
  python evaluate_image.py --model-path ../woV/Qwen3-8B --limit 20

  # 完整评测并保存结果
  python evaluate_image.py --model-path ../woV/Qwen3-8B --limit 50 --output results_text.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import random
import argparse
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# 复用 evaluate.py 中的评测组件
from evaluate import (
    BasePredictor,
    MockPredictor,
    FilePredictor,
    parse_model_output,
    Evaluator,
)

# 复用指标模块
from metrics import (
    compute_all_metrics,
    aggregate_metrics,
    edit_similarity,
    code_tokenize,
)


# ===========================================================================
# 1. 纯文本 TASK_PROMPT
# ===========================================================================

TASK_PROMPT_TEXT = """你是一名资深的代码补全助手，基于代码图谱 (Code Graph) 的文本表示来完成代码补全任务。

## 任务说明
你会收到一个代码子图的**文本序列化表示**，包含：
1. **节点列表**：子图中所有节点的详细信息
2. **边列表**：节点之间的结构关系、调用关系、导入关系

子图的**中心节点（目标节点）**的 text 和/或 comment 字段被 `[MASKED]` 替换，
你的任务是补全这些被屏蔽的字段。

## 代码图谱结构
图中包含以下节点类型：

### 节点类型及其关键字段
- **Repo**：仓库根节点。字段：repoName（仓库名）
- **Package**：目录/模块。字段：name（模块名）
- **File**：源代码文件。字段：fileName（文件名）, filePath（路径）, text（文件完整代码，可能被截断）
- **TextFile**：非代码文件（如 .md, .cfg, .txt）。字段：name（文件名）, path（路径）, text（文件内容，可能被截断）
- **Class**：类定义。字段：className（类名）, text（类体代码）, comment（docstring）
- **Function**：函数/方法定义。字段：name（函数名）, header（签名）, text（函数体代码）, comment（docstring）
- **Attribute**：类属性/成员变量。字段：name（属性名）, attributeType（类型标注）, comment（注释）
- **Lambda**：匿名lambda函数。字段：text（lambda 表达式）

### 边类型
- **contains**：结构包含关系（父节点 -> 子节点），如 Class contains Function、File contains Class
- **calls**：函数/方法调用关系（调用者 -> 被调用者）
- **imports**：模块导入关系

## 文本格式说明
子图的文本表示格式如下：

```
=== Code Graph Subgraph ===
Target Node: [MASKED] <节点类型> id=<id> name=<名称>

--- Nodes (总数) ---
[节点类型] id=<id>: <名称>
  <字段1>: <值>
  <字段2>: <值>

--- Edges (总数) ---
(<源id>:<源名称>) -[<边类型>]-> (<目标id>:<目标名称>)
```

注意事项：
- 目标节点（需要补全的节点）在节点列表中带有 `*** TARGET ***` 标记
- 被屏蔽的字段值为 `[MASKED]`
- 为控制上下文长度，File/TextFile 节点的 text 字段可能被截断，截断处标记 `... [TRUNCATED]`
- comment 字段值为 "null" 表示该节点没有注释/docstring，无需补全
- 代码风格应与子图中其他节点保持一致

## 你的任务
1. 仔细阅读子图的文本表示，理解各节点之间的结构关系和调用关系
2. 关注标记为 `[MASKED]` 的目标节点在子图中的位置和角色
3. 根据周围节点的信息（父类/文件、调用的函数、调用它的函数、相关属性等）推断补全内容
4. **将被 [MASKED] 标记的 text 和/或 comment 字段补全**

## 输出格式
请严格按照以下 JSON 格式输出补全结果，不要包含任何其他内容：

```json
{
  "text": "<完整的函数体/类体/代码实现>",
  "comment": "<完整的 docstring / 注释，如果原字段为 null 则输出 null>"
}
```

注意：
- text 字段应该包含完整的实现代码，包括函数签名或类定义行
- comment 字段补全 docstring，如果原图该字段是 "null" 则输出 null
- 代码风格应与子图中其他节点保持一致
- 补全的代码应该逻辑正确，能够通过静态分析
"""


# ===========================================================================
# 2. 子图文本序列化
# ===========================================================================

def _natural_identifier(node: dict) -> str:
    """返回节点的可读标识符（不含 id 前缀），与 CodeGraphRenderer 逻辑一致。"""
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


def _truncate_field(text: str, max_len: int) -> str:
    """截断过长文本并添加标记。"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return (
        text[:max_len]
        + f"\n... [TRUNCATED: {len(text)} chars total, showing first {max_len}]"
    )


def serialize_subgraph_as_text(
    sample: dict,
    max_file_text_len: int = 3000,
    max_function_text_len: int = 8000,
    max_class_text_len: int = 8000,
    max_comment_len: int = 2000,
    max_lambda_text_len: int = 8000,
) -> str:
    """将 benchmark 样本的 subgraph 序列化为结构化文本。

    Parameters
    ----------
    sample : dict
        Benchmark 样本，含 subgraph.nodes / subgraph.edges。
    max_file_text_len : int
        File/TextFile 节点的 text 字段最大字符数。
    max_function_text_len : int
        Function 节点的 text 字段最大字符数（种子节点不截断）。
    max_class_text_len : int
        Class 节点的 text 字段最大字符数（种子节点不截断）。
    max_comment_len : int
        comment 字段最大字符数。
    max_lambda_text_len : int
        Lambda 节点的 text 字段最大字符数。

    Returns
    -------
    str
        格式化的文本表示。
    """
    seed_id = sample["seed_node_id"]
    subgraph = sample["subgraph"]
    nodes: List[dict] = subgraph["nodes"]
    edges: List[dict] = subgraph["edges"]

    # 构建节点 id → node 索引
    node_by_id: Dict[int, dict] = {n["id"]: n for n in nodes}

    lines: List[str] = []

    # --- 头部：目标节点信息 ---
    seed_node = node_by_id.get(seed_id, {})
    seed_type = seed_node.get("nodeType", "?")
    seed_name = _natural_identifier(seed_node)
    masked_fields = sample.get("masked_fields", [])
    masked_str = ", ".join(masked_fields) if masked_fields else "none"

    lines.append("=== Code Graph Subgraph ===")
    lines.append(
        f"Target Node: [MASKED] {seed_type} id={seed_id} name={seed_name}"
    )
    lines.append(f"Masked fields: {masked_str}")
    lines.append("")

    # --- 节点列表 ---
    lines.append(f"--- Nodes ({len(nodes)} total) ---")
    lines.append("")

    # 按类型排序以提高可读性
    type_order = {
        "Repo": 0, "Package": 1, "File": 2, "TextFile": 3,
        "Class": 4, "Function": 5, "Attribute": 6, "Lambda": 7,
    }
    sorted_nodes = sorted(
        nodes, key=lambda n: (type_order.get(n.get("nodeType", ""), 99), n["id"])
    )

    for node in sorted_nodes:
        nid = node["id"]
        nt = node.get("nodeType", "?")
        name = _natural_identifier(node)
        is_target = nid == seed_id

        # 节点标题行
        target_marker = "  *** TARGET ***" if is_target else ""
        lines.append(f"[{nt}] id={nid}: {name}{target_marker}")

        # 按节点类型输出字段
        if nt == "Repo":
            # Repo 通常只有 repoName，已显示在标题中
            pass

        elif nt == "Package":
            # Package 只有 name
            pass

        elif nt == "File":
            text_val = node.get("text", "")
            if text_val:
                if is_target:
                    # 种子节点不会是被屏蔽的 File（种子只能是 Function/Class）
                    lines.append(f"  text: {_truncate_field(text_val, max_file_text_len)}")
                else:
                    lines.append(f"  text: {_truncate_field(text_val, max_file_text_len)}")

        elif nt == "TextFile":
            text_val = node.get("text", "")
            if text_val:
                lines.append(f"  text: {_truncate_field(text_val, max_file_text_len)}")

        elif nt == "Class":
            comment_val = node.get("comment", "null")
            text_val = node.get("text", "")
            # comment
            if comment_val == "[MASKED]":
                lines.append("  comment: [MASKED]")
            elif comment_val and comment_val != "null":
                lines.append(f"  comment: {_truncate_field(str(comment_val), max_comment_len)}")
            else:
                lines.append("  comment: null")
            # text
            if text_val == "[MASKED]":
                lines.append("  text: [MASKED]")
            elif is_target:
                # 种子节点不截断（实际上就是 [MASKED]）
                lines.append(f"  text: {text_val}")
            else:
                lines.append(f"  text: {_truncate_field(text_val, max_class_text_len)}")

        elif nt == "Function":
            header_val = node.get("header", "")
            comment_val = node.get("comment", "null")
            text_val = node.get("text", "")
            # header
            if header_val:
                lines.append(f"  header: {header_val}")
            # comment
            if comment_val == "[MASKED]":
                lines.append("  comment: [MASKED]")
            elif comment_val and comment_val != "null":
                lines.append(f"  comment: {_truncate_field(str(comment_val), max_comment_len)}")
            else:
                lines.append("  comment: null")
            # text
            if text_val == "[MASKED]":
                lines.append("  text: [MASKED]")
            elif is_target:
                lines.append(f"  text: {text_val}")
            else:
                lines.append(f"  text: {_truncate_field(text_val, max_function_text_len)}")

        elif nt == "Attribute":
            attr_type = node.get("attributeType", "null")
            comment_val = node.get("comment", "null")
            if attr_type and attr_type != "null":
                lines.append(f"  attributeType: {attr_type}")
            if comment_val and comment_val != "null":
                lines.append(f"  comment: {_truncate_field(str(comment_val), max_comment_len)}")

        elif nt == "Lambda":
            text_val = node.get("text", "")
            if text_val:
                lines.append(f"  text: {_truncate_field(text_val, max_lambda_text_len)}")

        lines.append("")

    # --- 边列表 ---
    lines.append(f"--- Edges ({len(edges)} total) ---")
    lines.append("")

    # 按边类型分组排序
    edge_order = {"contains": 0, "calls": 1, "imports": 2}
    sorted_edges = sorted(
        edges, key=lambda e: (edge_order.get(e.get("edgeType", ""), 99), e["source"], e["target"])
    )

    for edge in sorted_edges:
        src = edge["source"]
        tgt = edge["target"]
        etype = edge.get("edgeType", "?")
        src_node = node_by_id.get(src, {})
        tgt_node = node_by_id.get(tgt, {})
        src_name = _natural_identifier(src_node) if src_node else str(src)
        tgt_name = _natural_identifier(tgt_node) if tgt_node else str(tgt)
        # 截断过长的名称
        if len(src_name) > 40:
            src_name = src_name[:37] + "..."
        if len(tgt_name) > 40:
            tgt_name = tgt_name[:37] + "..."
        lines.append(f"({src}:{src_name}) -[{etype}]-> ({tgt}:{tgt_name})")

    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# 3. 纯文本模型封装 (Qwen3-8B)
# ===========================================================================

class TextModel:
    """封装 Qwen3-8B 纯文本模型的加载与推理。

    与 VT/model.py 的 ForwardModel (Qwen3-VL) 不同：
    - 使用 AutoModelForCausalLM + AutoTokenizer (而非 Qwen3VLForConditionalGeneration + AutoProcessor)
    - 消息 content 为纯字符串 (而非 [{type: "text", ...}, {type: "image", ...}])
    - 无需 qwen-vl-utils / process_vision_info
    - 使用采样解码 (temperature=0.6, top_p=0.95) 而非贪婪解码
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 8000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # 延迟导入，避免非 GPU 环境下的加载失败
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Qwen3-8B from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        print(f"  Model loaded: {type(self.model).__name__}")
        print(f"  Device: {self.model.device}")

    def generate(self, messages: List[Dict[str, str]], verbose: bool = True) -> str:
        """将纯文本 messages 送入 Qwen3-8B 并返回解码文本。

        Parameters
        ----------
        messages : list[dict]
            标准 ChatML 格式消息列表: {"role": "...", "content": "..."}
        verbose : bool
            是否打印推理详情。

        Returns
        -------
        str
            模型生成的文本。
        """
        # Step 1: 通过 chat template 生成文本 (关闭思考模式)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        if verbose:
            n_chars = len(text)
            print(f"  [TextModel] prompt={n_chars} chars ({n_chars//4}~ tokens)")

        # Step 2: tokenize 并移至模型设备
        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = inputs.to(self.model.device)

        if verbose:
            print(f"  [TextModel] input_ids shape={inputs.input_ids.shape}")

        # Step 3: 生成（采样模式，参考 generation_config.json）
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=True,
        )

        # Step 4: 截取新生成的 token
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Step 5: 解码
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result = output_text[0]

        if verbose:
            n_new = len(generated_ids_trimmed[0])
            preview = result[:300].replace("\n", "\\n")
            print(
                f"  [TextModel] generated {n_new} new tokens, "
                f"output preview: {preview}..."
            )

        return result


# ===========================================================================
# 4. 文本模式预测器
# ===========================================================================

class TextModelPredictor(BasePredictor):
    """纯文本模型预测器 — 将子图序列化为文本，调用 Qwen3-8B 进行补全。

    不需要 Graphviz、CodeGraphRenderer、OCRCoderIQ、ContextCompressor。
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str = "./eval_cache",
        max_new_tokens: int = 8000,
        # 截断参数
        max_file_text_len: int = 3000,
        max_function_text_len: int = 8000,
        max_class_text_len: int = 8000,
        max_comment_len: int = 2000,
        max_lambda_text_len: int = 8000,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        # 截断参数
        self.max_file_text_len = max_file_text_len
        self.max_function_text_len = max_function_text_len
        self.max_class_text_len = max_class_text_len
        self.max_comment_len = max_comment_len
        self.max_lambda_text_len = max_lambda_text_len

        # 缓存目录
        self._text_cache_dir = os.path.join(cache_dir, "text_representations")
        os.makedirs(self._text_cache_dir, exist_ok=True)

        # 延迟加载
        self._model: Optional[TextModel] = None

    def _ensure_model_loaded(self) -> None:
        """按需加载模型。"""
        if self._model is None:
            print("  [TextModelPredictor] Loading model (text-only mode) ...")
            self._model = TextModel(
                model_path=self.model_path,
                max_new_tokens=self.max_new_tokens,
            )

    def _get_text_cache_path(self, sample_id: str) -> str:
        """返回序列化文本的缓存文件路径。"""
        safe_name = sample_id.replace("/", "_").replace(":", "_").replace("#", "_")
        return os.path.join(self._text_cache_dir, f"{safe_name}.txt")

    def _build_text_representation(self, sample: dict) -> str:
        """将样本的子图序列化为文本（带缓存）。"""
        sample_id = sample["sample_id"]
        cache_path = self._get_text_cache_path(sample_id)

        # 命中缓存
        if os.path.isfile(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                cached = fh.read()
            if cached:
                return cached

        # 未命中 — 序列化并写入缓存
        text_repr = serialize_subgraph_as_text(
            sample,
            max_file_text_len=self.max_file_text_len,
            max_function_text_len=self.max_function_text_len,
            max_class_text_len=self.max_class_text_len,
            max_comment_len=self.max_comment_len,
            max_lambda_text_len=self.max_lambda_text_len,
        )

        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(text_repr)

        return text_repr

    def predict(self, sample: dict) -> str:
        self._ensure_model_loaded()

        # 1. 构建文本表示
        text_repr = self._build_text_representation(sample)

        # 2. 组装纯文本消息 (Qwen3 ChatML 格式)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": TASK_PROMPT_TEXT},
            {"role": "user", "content": text_repr},
        ]

        # 3. 推理
        assert self._model is not None
        response = self._model.generate(messages)

        if response:
            preview = response[:200].replace("\n", "\\n")
            print(
                f"  [TextModelPredictor] raw response "
                f"({len(response)} chars): {preview}..."
            )
        else:
            print("  [TextModelPredictor] WARNING: response is None or empty!")
        return response or ""


# ===========================================================================
# 5. CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CGM Benchmark — 纯文本评估引擎 (Qwen3-8B)"
    )

    # 数据
    parser.add_argument(
        "--benchmark",
        default=os.path.join(os.path.dirname(__file__), "benchmark_data.json"),
        help="Path to benchmark_data.json",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max samples to evaluate"
    )

    # 预测模式
    pred_group = parser.add_mutually_exclusive_group()
    pred_group.add_argument(
        "--mock",
        action="store_true",
        help="Use MockPredictor for pipeline testing",
    )
    pred_group.add_argument(
        "--predictions",
        default=None,
        help="Path to pre-saved predictions JSON (key=sample_id, value=raw_output)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to Qwen3-8B model for text-only inference",
    )

    # Mock 参数
    parser.add_argument(
        "--noise", type=float, default=0.2,
        help="Noise level for MockPredictor (0.0-1.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # 模型参数
    parser.add_argument(
        "--max-new-tokens", type=int, default=8000,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--max-file-text-len", type=int, default=3000,
        help="Max chars for File/TextFile node text",
    )
    parser.add_argument(
        "--max-function-text-len", type=int, default=8000,
        help="Max chars for Function node text",
    )
    parser.add_argument(
        "--max-class-text-len", type=int, default=8000,
        help="Max chars for Class node text",
    )
    parser.add_argument(
        "--max-comment-len", type=int, default=2000,
        help="Max chars for comment/docstring fields",
    )
    parser.add_argument(
        "--max-lambda-text-len", type=int, default=8000,
        help="Max chars for Lambda node text",
    )

    # 输出
    parser.add_argument(
        "--output", default=None, help="Save detailed results JSON"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-sample progress"
    )
    parser.add_argument(
        "--cache-dir", default="./eval_cache",
        help="Cache directory for serialized text",
    )

    args = parser.parse_args()

    # ---- 选择预测器 ----
    if args.mock:
        predictor = MockPredictor(noise_level=args.noise, seed=args.seed)
        print(f"Using MockPredictor (noise={args.noise})")
    elif args.predictions:
        predictor = FilePredictor(args.predictions)
        print(f"Using FilePredictor ({args.predictions})")
    elif args.model_path:
        predictor = TextModelPredictor(
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            max_new_tokens=args.max_new_tokens,
            max_file_text_len=args.max_file_text_len,
            max_function_text_len=args.max_function_text_len,
            max_class_text_len=args.max_class_text_len,
            max_comment_len=args.max_comment_len,
            max_lambda_text_len=args.max_lambda_text_len,
        )
        print(f"Using TextModelPredictor (text-only, {args.model_path})")
    else:
        # 默认: mock 模式
        predictor = MockPredictor(noise_level=0.2, seed=args.seed)
        print("Using MockPredictor (default, noise=0.2)")

    # ---- 运行评测 ----
    evaluator = Evaluator(
        benchmark_path=args.benchmark,
        predictor=predictor,
    )
    evaluator.load(limit=args.limit)
    evaluator.run()

    # ---- 输出 ----
    report = evaluator.report()
    print()
    print(report)

    if args.output:
        evaluator.save_results(args.output)


if __name__ == "__main__":
    main()
