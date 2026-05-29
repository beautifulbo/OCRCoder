"""
CGM Benchmark 评测指标。

纯 Python 标准库实现，零外部依赖，涵盖:
  - 硬逻辑: 编辑距离、Token 级编辑距离、Exact Match
  - N-gram: BLEU 变体
  - 语义: TF-IDF 向量 + 余弦相似度
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 代码分词器
# ---------------------------------------------------------------------------

# 代码中常见的分隔模式
_CODE_TOKEN_PATTERN = re.compile(
    r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+|[^\s\w]|\w+",
    re.UNICODE,
)


def code_tokenize(text: str) -> List[str]:
    """
    将代码字符串分词为 token 列表。

    使用启发式规则处理:
    - snake_case → ['snake', 'case']
    - camelCase  → ['camel', 'Case']
    - PascalCase → ['Pascal', 'Case']
    - 保留运算符和标点作为独立 token
    - 数字作为独立 token
    """
    if not text:
        return []

    tokens: List[str] = []
    for match in _CODE_TOKEN_PATTERN.finditer(text):
        raw = match.group(0)
        # 进一步拆分 snake_case
        if "_" in raw and len(raw) > 1:
            parts = raw.split("_")
            tokens.extend(p for p in parts if p)
        else:
            tokens.append(raw)
    return tokens


# ---------------------------------------------------------------------------
# 1. 编辑距离 (Levenshtein)
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    纯 Python O(n*m) Levenshtein 编辑距离。
    对长字符串使用空间优化版本（两行 DP）。
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    # 空间优化：只保留两行
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            if c1 == c2:
                curr.append(prev[j - 1])
            else:
                curr.append(1 + min(prev[j], curr[j - 1], prev[j - 1]))
        prev = curr

    return prev[-1]


def edit_similarity(s1: str, s2: str) -> float:
    """
    归一化编辑相似度: 1 - edit_distance / max(len(s1), len(s2))

    值域 [0, 1]，1 表示完全相同。
    两个都为空字符串时返回 1.0。
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - dist / max_len


def token_edit_similarity(text1: str, text2: str) -> float:
    """
    Token 级编辑相似度: 先分词为 token 列表，再在 token 序列上计算编辑距离。
    """
    t1 = code_tokenize(text1)
    t2 = code_tokenize(text2)
    if not t1 and not t2:
        return 1.0
    max_len = max(len(t1), len(t2))
    if max_len == 0:
        return 1.0
    # 在 token ID 序列上计算编辑距离
    dist = levenshtein_distance(t1, t2)
    return 1.0 - dist / max_len


# ---------------------------------------------------------------------------
# 2. Exact Match
# ---------------------------------------------------------------------------

def exact_match(pred: str, gt: str) -> float:
    """严格完全匹配，返回 1.0 或 0.0。"""
    return 1.0 if pred == gt else 0.0


def normalized_exact_match(pred: str, gt: str) -> float:
    """去除首尾空白后匹配。"""
    return 1.0 if pred.strip() == gt.strip() else 0.0


# ---------------------------------------------------------------------------
# 3. SequenceMatcher (difflib)
# ---------------------------------------------------------------------------

def sequence_ratio(pred: str, gt: str) -> float:
    """difflib.SequenceMatcher 的 ratio()，基于最长公共子序列。"""
    if not pred and not gt:
        return 1.0
    return SequenceMatcher(None, pred, gt).ratio()


# ---------------------------------------------------------------------------
# 4. BLEU-like N-gram 指标
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def code_bleu(pred: str, gt: str, max_n: int = 4, smooth: bool = True) -> float:
    """
    代码专用的 BLEU 变体。

    - 使用 code_tokenize 分词
    - 计算 n=1..max_n 的 n-gram 精度
    - 加入简短惩罚 (brevity penalty)
    - smooth=True 时对小 n-gram 做加 1 平滑
    """
    pred_tokens = code_tokenize(pred)
    gt_tokens = code_tokenize(gt)

    if not pred_tokens or not gt_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(pred_tokens, n)
        gt_ngrams = _ngrams(gt_tokens, n)

        if not pred_ngrams:
            precisions.append(0.0)
            continue

        match_count = 0
        total_count = 0
        for ngram, count in pred_ngrams.items():
            total_count += count
            match_count += min(count, gt_ngrams.get(ngram, 0))

        if smooth and match_count == 0:
            match_count = 1  # 平滑

        precisions.append(match_count / max(total_count, 1))

    # 几何平均
    geo_mean = 1.0
    for p in precisions:
        if p > 0:
            geo_mean *= p
        else:
            geo_mean *= 1e-10  # 避免完全为 0
    geo_mean = geo_mean ** (1.0 / max_n)

    # 简短惩罚
    bp = min(1.0, math.exp(1.0 - len(gt_tokens) / max(len(pred_tokens), 1)))

    return bp * geo_mean


# ---------------------------------------------------------------------------
# 5. TF-IDF 语义相似度
# ---------------------------------------------------------------------------

def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """计算词频 (term frequency)。"""
    counter = Counter(tokens)
    total = len(tokens) or 1
    return {word: count / total for word, count in counter.items()}


def _compute_idf(corpus: List[List[str]]) -> Dict[str, float]:
    """计算逆文档频率 (inverse document frequency)。"""
    num_docs = len(corpus)
    df: Counter = Counter()
    for tokens in corpus:
        df.update(set(tokens))
    return {
        word: math.log((num_docs + 1) / (count + 1)) + 1.0
        for word, count in df.items()
    }


def _tfidf_vector(
    tokens: List[str], idf: Dict[str, float]
) -> Dict[str, float]:
    """计算 TF-IDF 向量。"""
    tf = _compute_tf(tokens)
    vec: Dict[str, float] = {}
    for word, tf_val in tf.items():
        vec[word] = tf_val * idf.get(word, 1.0)
    return vec


def _cosine_similarity(
    vec1: Dict[str, float], vec2: Dict[str, float]
) -> float:
    """计算两个稀疏向量的余弦相似度。"""
    dot = sum(vec1.get(k, 0.0) * vec2.get(k, 0.0) for k in set(vec1) | set(vec2))
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values())) or 1e-10
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values())) or 1e-10
    return dot / (norm1 * norm2)


def semantic_similarity(
    pred: str, gt: str, corpus: Optional[List[str]] = None
) -> float:
    """
    基于 TF-IDF 的语义相似度。

    如果提供 corpus（所有样本的 GT），用语料级 IDF；
    否则用单样本的 TF 作为退化的近似。

    Parameters
    ----------
    pred : str
        模型预测的代码
    gt : str
        真实代码
    corpus : list[str] or None
        所有样本的 GT 列表，用于计算 IDF。None 则退化为单样本 TF 余弦相似度。
    """
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0

    pred_tokens = code_tokenize(pred)
    gt_tokens = code_tokenize(gt)

    if corpus is not None:
        all_tokens = [code_tokenize(doc) for doc in corpus]
        idf = _compute_idf(all_tokens)
    else:
        idf = _compute_idf([pred_tokens, gt_tokens])

    vec1 = _tfidf_vector(pred_tokens, idf)
    vec2 = _tfidf_vector(gt_tokens, idf)

    return _cosine_similarity(vec1, vec2)


# ---------------------------------------------------------------------------
# 6. 综合评测
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred: str,
    gt: str,
    corpus_for_tfidf: Optional[List[str]] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """
    计算所有指标，返回字典。

    Parameters
    ----------
    pred : str
        模型预测
    gt : str
        真实值
    corpus_for_tfidf : list[str] or None
        全局 GT 列表，用于 TF-IDF 的 IDF 计算
    prefix : str
        指标名前缀，如 "text_" 或 "comment_"
    """
    return {
        f"{prefix}exact_match": exact_match(pred, gt),
        f"{prefix}norm_exact_match": normalized_exact_match(pred, gt),
        f"{prefix}edit_similarity": edit_similarity(pred, gt),
        f"{prefix}token_edit_similarity": token_edit_similarity(pred, gt),
        f"{prefix}sequence_ratio": sequence_ratio(pred, gt),
        f"{prefix}code_bleu": code_bleu(pred, gt),
        f"{prefix}semantic_similarity": semantic_similarity(
            pred, gt, corpus=corpus_for_tfidf
        ),
    }


# ---------------------------------------------------------------------------
# 7. 统计聚合
# ---------------------------------------------------------------------------

def aggregate_metrics(
    per_sample: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    对多个样本的指标做聚合统计: mean, std, min, max, median。

    返回 {metric_name: {"mean": ..., "std": ..., "min": ..., "max": ..., "median": ...}}
    """
    if not per_sample:
        return {}

    keys = list(per_sample[0].keys())
    aggregated: Dict[str, Dict[str, float]] = {}

    for key in keys:
        values = sorted(m[key] for m in per_sample)
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2.0
        aggregated[key] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": values[0],
            "max": values[-1],
            "median": median,
        }
    return aggregated


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 快速自测
    gt = (
        "def add(a: int, b: int) -> int:\n"
        '    """Add two integers."""\n'
        "    return a + b\n"
    )

    # 完全正确
    print("=== Exact Match ===")
    print("  EM:", exact_match(gt, gt))
    print("  ES:", edit_similarity(gt, gt))

    # 部分正确
    pred1 = "def add(a, b):\n    return a + b\n"
    print("\n=== Partial Match ===")
    for k, v in compute_all_metrics(pred1, gt).items():
        print(f"  {k}: {v:.3f}")

    # 完全不同
    pred2 = "def subtract(x, y):\n    return x - y\n"
    print("\n=== Different ===")
    for k, v in compute_all_metrics(pred2, gt).items():
        print(f"  {k}: {v:.3f}")

    print("\nAll metrics OK.")
