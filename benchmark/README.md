# CGM Benchmark — 代码图谱补全评测系统

基于 [swe-bench-lite](../code_graph_data/swe-bench-lite/) 代码图谱数据集，评估模型在子图上下文中对缺失代码的补全能力。

## 目录

- [快速开始](#快速开始)
- [数据准备 (prepare_data.py)](#数据准备-prepare_datapy)
- [运行评测 (evaluate.py)](#运行评测-evaluatepy)
- [评测指标 (metrics.py)](#评测指标-metricspy)
- [Schema 解析 (parse_schema.py)](#schema-解析-parse_schemapy)
- [依赖安装](#依赖安装)
- [数据格式说明](#数据格式说明)

---

## 快速开始

```bash
# 1. 安装依赖（Mock 模式零外部依赖，可跳过此步）
pip install -r requirements-full.txt

# 2. 准备评测数据（从 swe-bench-lite 生成）
python prepare_data.py \
  --graph-dir ../code_graph_data/swe-bench-lite \
  --samples-per-graph 4 \
  --output benchmark_data.json

# 3. 测试评测管线（Mock 模式，无需 GPU）
python evaluate.py --mock --limit 20

# 4. 真实模型评测（需 GPU）
python evaluate.py \
  --model-path /path/to/Qwen3-VL-2B-Instruct \
  --graph-dir ../code_graph_data/swe-bench-lite \
  --limit 50 \
  --output results.json
```

---

## 数据准备 (prepare_data.py)

从 swe-bench-lite 的 `.graph.json` 文件生成 benchmark 数据。

### 处理流程

```
.graph.json 文件
  └─ 加载图 → 随机选取 Function / Class 种子节点
       └─ 以种子为中心 k-hop BFS 采样子图 (max_nodes 上限)
            └─ 屏蔽种子节点的 text / comment 字段 → [MASKED]
                 └─ 输出 input (屏蔽后) + ground_truth (原始值)
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--graph-dir` | str | `../code_graph_data/swe-bench-lite` | `.graph.json` 文件目录 |
| `--output` | str | `benchmark_data.json` | 输出 JSON 路径 |
| `--samples-per-graph` | int | `4` | 每个图文件中采样的种子节点数 |
| `--k-hops` | int | `2` | BFS 扩展跳数 |
| `--max-nodes` | int | `150` | 子图最大节点数 |
| `--seed` | int | `42` | 随机种子（保证可复现） |
| `--limit-graphs` | int | None | 限制处理的图文件数（用于测试） |
| `--min-text-len` | int | `50` | 种子节点 text 字段最小长度（过滤太短的节点） |

### 使用示例

```bash
# 生成完整 benchmark（274 个图，约 1078 个样本）
python prepare_data.py

# 快速测试：处理 5 个图，每个图 2 个样本
python prepare_data.py --limit-graphs 5 --samples-per-graph 2 --output test.json

# 调整子图大小
python prepare_data.py --max-nodes 200 --k-hops 3
```

### 输出格式

```json
[
  {
    "sample_id": "django#django#xxx.graph.json::seed_12345",
    "source_graph": "django#django#xxx.graph.json",
    "seed_node_id": 12345,
    "seed_node_type": "Function",
    "masked_fields": ["text", "comment"],
    "subgraph": {
      "nodes": [{ "id": 12345, "nodeType": "Function", "text": "[MASKED]", ... }, ...],
      "edges": [{ "edgeType": "calls", "source": 12345, "target": 678, ... }, ...]
    },
    "ground_truth": {
      "node_id": 12345,
      "text": "def foo():\n    return 42",
      "comment": "Returns the answer."
    }
  },
  ...
]
```

- `subgraph`：屏蔽后的输入子图，遵循原始 `{nodes, edges}` 格式，种子节点的被屏蔽字段为 `[MASKED]`
- `ground_truth`：仅包含种子节点的原始值，key 为被屏蔽的字段名

---

## 运行评测 (evaluate.py)

三种预测模式，通过互斥参数选择。

### 预测模式

| 模式 | 参数 | 说明 | 需要 GPU |
|------|------|------|----------|
| Mock | `--mock` | 在 GT 上注入可配置噪声，用于测试评测管线 | 否 |
| File | `--predictions FILE` | 从预存 JSON 文件读取预测 | 否 |
| Model | `--model-path PATH` | 调用真实 Qwen3-VL 模型推理 | 是 |

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--benchmark` | str | `benchmark_data.json` | Benchmark 数据路径 |
| `--limit` | int | None | 最多评测的样本数 |
| `--mock` | flag | False | 使用 MockPredictor |
| `--predictions` | str | None | 预存预测文件路径（key=sample_id） |
| `--model-path` | str | None | Qwen3-VL 模型路径（启用真实推理） |
| `--graph-dir` | str | `../code_graph_data/swe-bench-lite` | `.graph.json` 目录（`--model-path` 时需要） |
| `--noise` | float | `0.2` | Mock 噪声水平 (0.0=满分, 1.0=完全随机) |
| `--seed` | int | `42` | 随机种子 |
| `--output` | str | None | 保存详细评测结果 JSON |
| `--quiet` | flag | False | 抑制逐样本进度输出 |

### 使用示例

```bash
# === Mock 模式：快速测试评测管线 ===
python evaluate.py --mock --limit 50 --noise 0.15

# === File 模式：离线评测（推荐工作流） ===
# 第一步：生成预测文件（用模型或脚本）
python -c "
import json
predictions = {}
for sample in benchmark:
    predictions[sample['sample_id']] = model.predict(sample)
json.dump(predictions, open('preds.json', 'w'))
"
# 第二步：评测
python evaluate.py --predictions preds.json --limit 200 --output results.json

# === Model 模式：端到端评测 ===
python evaluate.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --graph-dir ../code_graph_data/swe-bench-lite \
  --limit 20 \
  --output results.json
```

### 评测报告内容

运行结束后输出：

- **TEXT / COMMENT 字段各 7 个指标**：mean, std, min, max, median
- **按节点类型拆分**：Class vs Function 的 text_edit_similarity
- **分位数分布**：text_edit_similarity 的 P10/P25/P50/P75/P90/P95/P99

---

## 评测指标 (metrics.py)

纯 Python 标准库实现，零外部依赖。共 7 个指标：

| 指标 | 函数 | 值域 | 说明 |
|------|------|------|------|
| Exact Match | `exact_match()` | {0, 1} | 预测与 GT 逐字符完全相等 |
| Norm EM | `normalized_exact_match()` | {0, 1} | 去首尾空白后匹配 |
| Edit Similarity | `edit_similarity()` | [0, 1] | 1 − Levenshtein / max(len) |
| Token Edit Sim | `token_edit_similarity()` | [0, 1] | 先 code_tokenize 分词，再 Token 级编辑距离 |
| Sequence Ratio | `sequence_ratio()` | [0, 1] | difflib 最长公共子序列比 |
| CodeBLEU | `code_bleu()` | [0, 1] | n=1..4 gram + brevity penalty |
| Semantic Sim | `semantic_similarity()` | [0, 1] | TF-IDF + 余弦相似度（支持语料级 IDF） |

### 代码分词器 (`code_tokenize`)

专门针对代码特性设计：自动处理 `snake_case`、`camelCase`、`PascalCase`，保留运算符和标点作为独立 token。

```python
from metrics import code_tokenize, compute_all_metrics

tokens = code_tokenize("def getMaxValue(self, x):")
# ['def', 'get', 'Max', 'Value', '(', 'self', ',', 'x', ')', ':']

metrics = compute_all_metrics(pred_code, gt_code, prefix="text_")
# {"text_exact_match": 0.0, "text_edit_similarity": 0.85, ...}
```

---

## Schema 解析 (parse_schema.py)

解析 swe-bench-lite 中 `.graph.json` 的数据 schema，统计每种节点/边类型的字段及属性。

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--graph-dir` | str | `../code_graph_data/swe-bench-lite` | 图文件目录 |
| `--limit` | int | None | 扫描文件数上限 |
| `--output` | str | None | 保存 schema JSON 到文件 |
| `--sample-values` | int | `3` | 每个字段展示的样本值数量 |

### 使用示例

```bash
# 全量扫描，输出到终端 + schema.json
python parse_schema.py --output schema.json

# 快速预览（前 20 个文件）
python parse_schema.py --limit 20 --sample-values 2
```

---

## 依赖安装

| 文件 | 用途 |
|------|------|
| [requirements.txt](requirements.txt) | 文档型，各模式依赖说明 |
| [requirements-gpu.txt](requirements-gpu.txt) | GPU 推理额外依赖 |
| [requirements-full.txt](requirements-full.txt) | 一键安装全部 |

```bash
# Mock 评测模式：零外部依赖，Python 3.10+ 标准库即可

# 完整 GPU 推理模式：
pip install -r requirements-full.txt
apt install graphviz     # graphviz 系统二进制
```

### 依赖清单（GPU 模式）

```
torch>=2.0.0
transformers>=4.45.0       # Qwen3VLForConditionalGeneration
accelerate>=0.20.0         # device_map="auto"
qwen-vl-utils>=0.0.6       # Qwen3-VL 图像处理
Pillow>=9.0.0              # ContextCompressor 图像渲染
numpy>=1.24.0              # AgentOCR 数值计算
graphviz>=0.20.0           # CodeGraphRenderer 子图渲染
networkx>=3.0              # BFS 子图采样
bitsandbytes>=0.41.0       # 量化（可选）
peft>=0.7.0                # LoRA（可选）
```

---

## 数据格式说明

### .graph.json 文件结构

由 swe-bench-lite 提供的代码图谱文件，遵循固定 schema（详见 `python parse_schema.py` 输出）：

```json
{
  "nodes": [
    {
      "id": 12345,
      "nodeType": "Function",
      "name": "get_user",
      "header": "def get_user(id: int) -> User:",
      "text": "def get_user(id: int) -> User:\n    return db.query(User).filter(...)",
      "comment": "Fetch a user by their ID.",
      "col": 4,
      "startLoc": 42,
      "endLoc": 50
    }
  ],
  "edges": [
    {
      "edgeType": "calls",
      "source": 12345,
      "target": 678
    }
  ]
}
```

**8 种节点类型**：`Repo`, `Package`, `File`, `TextFile`, `Class`, `Function`, `Attribute`, `Lambda`

**3 种边类型**：`contains`（结构包含）, `calls`（函数调用）, `imports`（模块导入）

> 注意：`comment` 字段使用字符串 `"null"` 表示无注释（非 JSON null）。

### Benchmark 样本结构

见上文[数据准备 — 输出格式](#输出格式)。

---

## 文件索引

```
tools/benchmark/
├── README.md                  # 本文件
├── prepare_data.py            # 数据准备脚本
├── evaluate.py                # 评测引擎
├── metrics.py                 # 评测指标（纯 Python）
├── parse_schema.py            # Schema 解析
├── stats.py                   # 数据统计（辅助）
├── requirements.txt           # 依赖说明
├── requirements-gpu.txt       # GPU 依赖
├── requirements-full.txt      # 完整依赖
└── benchmark_data.json        # 生成的评测数据（355 MB，1078 样本）
```
