"""
CGM Benchmark 评测引擎。

用法:
  # Mock 模式（测试评测管线）
  python evaluate.py --mock --limit 20

  # 仅计算指标（已有预测文件时）
  python evaluate.py --predictions predictions.json --limit 50

  # 完整评测（需 GPU + Qwen3-VL 模型）
  python evaluate.py --model-path ./Qwen3-VL-2B-Instruct --limit 20
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

from metrics import (
    compute_all_metrics,
    aggregate_metrics,
    edit_similarity,
    code_tokenize,
)


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\"text\"[^{}]*\}", re.DOTALL)


def parse_model_output(raw: str) -> Dict[str, Optional[str]]:
    """
    从模型输出中解析 text 和 comment 字段。

    支持两种格式:
    1. ```json { "text": "...", "comment": "..." } ```
    2. 裸 JSON 对象 { "text": "...", "comment": "..." }
    """
    text_val: Optional[str] = None
    comment_val: Optional[str] = None

    # 优先匹配 ```json ... ``` 代码块
    m = _JSON_BLOCK_RE.search(raw)
    json_str = m.group(1) if m else None

    # 其次匹配裸 JSON 对象
    if json_str is None:
        m = _JSON_OBJECT_RE.search(raw)
        json_str = m.group(0) if m else raw

    try:
        obj = json.loads(json_str)
        text_val = obj.get("text")
        comment_val = obj.get("comment")
    except (json.JSONDecodeError, TypeError):
        # 回退：把整个输出当 text
        text_val = raw.strip()

    return {"text": text_val, "comment": comment_val}


# ---------------------------------------------------------------------------
# 预测器接口
# ---------------------------------------------------------------------------

class BasePredictor:
    """预测器基类。"""

    def predict(self, sample: dict) -> str:
        """给定一个 benchmark 样本，返回模型输出的原始字符串。"""
        raise NotImplementedError


class MockPredictor(BasePredictor):
    """
    模拟预测器 — 用于测试评测管线。

    策略: 在 ground_truth.text 基础上随机注入噪声:
    - noise_level=0.0:   返回原始 GT（满分）
    - noise_level=0.3:   删除/插入/替换 30% 字符
    - noise_level=1.0:   完全随机输出
    """

    def __init__(self, noise_level: float = 0.2, seed: int = 42):
        self.noise_level = noise_level
        random.seed(seed)

    def _mutate(self, text: str) -> str:
        if not text:
            return text
        chars = list(text)
        n_changes = max(1, int(len(chars) * self.noise_level))
        for _ in range(n_changes):
            op = random.random()
            idx = random.randrange(len(chars))
            if op < 0.33 and len(chars) > 1:
                # 删除
                del chars[idx]
            elif op < 0.66:
                # 替换
                chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz_(){}[] ")
            else:
                # 插入
                chars.insert(idx, random.choice(" \n\t"))
        return "".join(chars)

    def predict(self, sample: dict) -> str:
        gt = sample["ground_truth"]
        text_val = gt.get("text", "")
        comment_val = gt.get("comment")

        mutated_text = self._mutate(text_val)

        if comment_val:
            mutated_comment = self._mutate(comment_val)
            output = json.dumps(
                {"text": mutated_text, "comment": mutated_comment},
                ensure_ascii=False,
            )
        else:
            output = json.dumps({"text": mutated_text}, ensure_ascii=False)

        return output


class FilePredictor(BasePredictor):
    """从预存文件读取预测结果。"""

    def __init__(self, predictions_path: str):
        with open(predictions_path, "r", encoding="utf-8") as fh:
            self.predictions: Dict[str, str] = json.load(fh)
        # key: sample_id → raw output

    def predict(self, sample: dict) -> str:
        sid = sample["sample_id"]
        return self.predictions.get(sid, "")


class ModelPredictor(BasePredictor):
    """
    真实模型预测器 — 封装 OCRCoderIQ + ForwardModel。

    需要 GPU 环境和完整依赖（transformers, Graphviz 等）。
    """

    def __init__(
        self,
        model_path: str,
        graph_dir: str,
        cache_dir: str = "./eval_cache",
        max_new_tokens: int = 8000,
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "graph_images"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "context_images"), exist_ok=True)

        # 延迟导入，避免非 GPU 环境下的加载失败
        # 将 VT 目录加入 sys.path，以便直接从该目录导入模块 (model, renderer 等)
        _vt_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "VT")
        )
        if _vt_dir not in sys.path:
            sys.path.insert(0, _vt_dir)
        from model import ForwardModel, OCRCoderIQ
        from renderer import CodeGraphRenderer
        from AgentOCR.agentocr.context_compressor import (
            ContextCompressor,
            CompressorConfig,
        )

        self._ForwardModel = ForwardModel
        self._OCRCoderIQ = OCRCoderIQ
        self._CodeGraphRenderer = CodeGraphRenderer
        self._ContextCompressor = ContextCompressor
        self._CompressorConfig = CompressorConfig

        self.model_path = model_path
        self.graph_dir = graph_dir
        self.max_new_tokens = max_new_tokens

        # 延迟初始化（在第一个样本时加载模型）
        self._model: Any = None
        self._coder: Any = None
        self._current_graph: Optional[str] = None

    def _ensure_model_loaded(self, graph_file: str):
        """按需加载模型和图渲染器。"""
        if self._model is None:
            print("  [ModelPredictor] Loading model ...")
            self._model = self._ForwardModel(
                model_path=self.model_path,
                max_new_tokens=self.max_new_tokens,
            )

        if graph_file != self._current_graph:
            graph_path = os.path.join(self.graph_dir, graph_file)
            if not os.path.isfile(graph_path):
                raise FileNotFoundError(f"Graph not found: {graph_path}")
            renderer = self._CodeGraphRenderer(graph_path)
            # 评测模式下传空 context，ContextCompressor 会返回空图片列表
            context_renderer = self._ContextCompressor(
                self._CompressorConfig.balanced()
            )
            self._coder = self._OCRCoderIQ(
                context=[],
                window_size=5,
                file_path=os.path.join(self.cache_dir, "graph_images"),
                graph_render=renderer,
                context_render=context_renderer,
                session_id=f"eval_{os.path.basename(graph_file)}",
                model=self._model,
            )
            self._current_graph = graph_file

    def predict(self, sample: dict) -> str:
        graph_file = sample["source_graph"]
        seed_id = sample["seed_node_id"]

        self._ensure_model_loaded(graph_file)

        response = self._coder.generateVQ(
            seed_id=seed_id,
            context=[],
            user_query="Complete the masked node.",
        )
        if response:
            preview = response[:200].replace("\n", "\\n")
            print(f"  [ModelPredictor] raw response ({len(response)} chars): {preview}...")
        else:
            print(f"  [ModelPredictor] WARNING: response is None or empty!")
        return response or ""


# ---------------------------------------------------------------------------
# 评测器
# ---------------------------------------------------------------------------

class Evaluator:
    """加载 benchmark 数据，运行预测，计算指标。"""

    def __init__(
        self,
        benchmark_path: str,
        predictor: BasePredictor,
    ):
        self.benchmark_path = benchmark_path
        self.predictor = predictor

        self.samples: List[dict] = []
        self.results: List[dict] = []

    def load(self, limit: Optional[int] = None):
        """加载 benchmark 数据。"""
        print(f"Loading benchmark from {self.benchmark_path} ...")
        with open(self.benchmark_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if limit:
            data = data[:limit]
        self.samples = data
        print(f"Loaded {len(self.samples)} samples.")

    def run(self):
        """
        逐样本评测:
        1. 收集所有 GT text，构建 TF-IDF 语料
        2. 对每个样本运行预测 + 计算指标
        """
        # 构建全局 TF-IDF 语料（所有 GT text）
        all_gt_texts = [
            s["ground_truth"].get("text", "") or ""
            for s in self.samples
        ]

        print(f"Running evaluation on {len(self.samples)} samples ...")
        start_time = time.time()

        for i, sample in enumerate(self.samples):
            sid = sample["sample_id"]
            gt = sample["ground_truth"]
            masked_fields = sample["masked_fields"]

            # 预测
            raw_pred = self.predictor.predict(sample)
            parsed = parse_model_output(raw_pred)

            # 计分: text 字段
            pred_text = parsed.get("text") or ""
            gt_text = gt.get("text") or ""

            text_metrics = compute_all_metrics(
                pred_text, gt_text,
                corpus_for_tfidf=all_gt_texts,
                prefix="text_",
            )

            # 计分: comment 字段（仅在被屏蔽时）
            comment_metrics: Dict[str, float] = {}
            if "comment" in masked_fields:
                pred_comment = parsed.get("comment") or ""
                gt_comment = gt.get("comment") or ""
                comment_metrics = compute_all_metrics(
                    pred_comment, gt_comment,
                    prefix="comment_",
                )

            # 汇总
            all_metrics = {**text_metrics, **comment_metrics}
            self.results.append({
                "sample_id": sid,
                "seed_node_type": sample["seed_node_type"],
                "masked_fields": masked_fields,
                "metrics": all_metrics,
                "pred_text_len": len(pred_text),
                "gt_text_len": len(gt_text),
                "raw_pred_preview": raw_pred[:200],
            })

            # 进度
            if (i + 1) % 10 == 0 or i == len(self.samples) - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 1)
                eta = (len(self.samples) - i - 1) / max(rate, 0.001)
                print(f"  [{i+1}/{len(self.samples)}] "
                      f"rate={rate:.1f}/s  "
                      f"edit_sim={all_metrics.get('text_edit_similarity', 0):.3f}  "
                      f"ETA={eta:.0f}s")

        total_time = time.time() - start_time
        print(f"Done in {total_time:.1f}s ({total_time/len(self.samples):.1f}s/sample)")

    def report(self) -> str:
        """生成评测报告。"""
        if not self.results:
            return "No results to report."

        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("CGM BENCHMARK EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Samples evaluated: {len(self.results)}")
        lines.append("")

        # 分离 text 和 comment 指标
        text_metrics_list = [
            {k: v for k, v in r["metrics"].items() if k.startswith("text_")}
            for r in self.results
            if any(k.startswith("text_") for k in r["metrics"])
        ]
        comment_metrics_list = [
            {k: v for k, v in r["metrics"].items() if k.startswith("comment_")}
            for r in self.results
            if any(k.startswith("comment_") for k in r["metrics"])
        ]

        # --- Text 指标 ---
        if text_metrics_list:
            agg_text = aggregate_metrics(text_metrics_list)
            lines.append("-" * 50)
            lines.append("TEXT FIELD METRICS")
            lines.append("-" * 50)
            lines.append(f"{'Metric':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
            lines.append("-" * 70)
            for metric, stats in agg_text.items():
                short_name = metric.replace("text_", "")
                lines.append(
                    f"{short_name:<30} "
                    f"{stats['mean']:8.4f} "
                    f"{stats['std']:8.4f} "
                    f"{stats['min']:8.4f} "
                    f"{stats['max']:8.4f} "
                    f"{stats['median']:8.4f}"
                )

        # --- Comment 指标 ---
        if comment_metrics_list:
            agg_comment = aggregate_metrics(comment_metrics_list)
            lines.append("")
            lines.append("-" * 50)
            lines.append("COMMENT FIELD METRICS")
            lines.append("-" * 50)
            lines.append(f"{'Metric':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
            lines.append("-" * 70)
            for metric, stats in agg_comment.items():
                short_name = metric.replace("comment_", "")
                lines.append(
                    f"{short_name:<30} "
                    f"{stats['mean']:8.4f} "
                    f"{stats['std']:8.4f} "
                    f"{stats['min']:8.4f} "
                    f"{stats['max']:8.4f} "
                    f"{stats['median']:8.4f}"
                )

        # --- 按节点类型拆分 ---
        lines.append("")
        lines.append("-" * 50)
        lines.append("BY SEED NODE TYPE (text_edit_similarity)")
        lines.append("-" * 50)
        by_type: Dict[str, List[float]] = defaultdict(list)
        for r in self.results:
            nt = r["seed_node_type"]
            if "text_edit_similarity" in r["metrics"]:
                by_type[nt].append(r["metrics"]["text_edit_similarity"])
        for nt in sorted(by_type.keys()):
            vals = by_type[nt]
            mean = sum(vals) / len(vals)
            lines.append(f"  {nt:<12}: mean={mean:.4f}  n={len(vals)}")

        # --- 分位数分布 ---
        if text_metrics_list:
            lines.append("")
            lines.append("-" * 50)
            lines.append("TEXT EDIT SIMILARITY — PERCENTILE DISTRIBUTION")
            lines.append("-" * 50)
            es_vals = sorted(
                r["metrics"].get("text_edit_similarity", 0)
                for r in self.results
                if "text_edit_similarity" in r["metrics"]
            )
            n = len(es_vals)
            for p in [10, 25, 50, 75, 90, 95, 99]:
                idx = min(int(n * p / 100), n - 1)
                lines.append(f"  P{p:>2}: {es_vals[idx]:.4f}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def save_results(self, output_path: str):
        """保存详细结果到 JSON。"""
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(self.results, fh, ensure_ascii=False, indent=2)
        print(f"Detailed results saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CGM Benchmark Evaluator")

    # 数据
    parser.add_argument(
        "--benchmark",
        default=os.path.join(os.path.dirname(__file__), "benchmark_data.json"),
        help="Path to benchmark_data.json",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max samples to evaluate")

    # 预测模式
    pred_group = parser.add_mutually_exclusive_group()
    pred_group.add_argument(
        "--mock", action="store_true",
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
        help="Path to Qwen3-VL model (enables real model inference)",
    )
    parser.add_argument(
        "--graph-dir",
        default="/home/10358884/10358884/tools/code_graph_data/swe-bench-lite",
        help="Path to .graph.json files (needed for --model-path)",
    )

    # Mock 参数
    parser.add_argument("--noise", type=float, default=0.2,
                        help="Noise level for MockPredictor (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # 输出
    parser.add_argument("--output", default=None, help="Save detailed results JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-sample progress")

    args = parser.parse_args()

    # ---- 选择预测器 ----
    if args.mock:
        predictor = MockPredictor(noise_level=args.noise, seed=args.seed)
        print(f"Using MockPredictor (noise={args.noise})")
    elif args.predictions:
        predictor = FilePredictor(args.predictions)
        print(f"Using FilePredictor ({args.predictions})")
    elif args.model_path:
        predictor = ModelPredictor(
            model_path=args.model_path,
            graph_dir=args.graph_dir,
        )
        print(f"Using ModelPredictor ({args.model_path})")
    else:
        # 默认: mock 模式
        predictor = MockPredictor(noise_level=0.2, seed=args.seed)
        print(f"Using MockPredictor (default, noise=0.2)")

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
