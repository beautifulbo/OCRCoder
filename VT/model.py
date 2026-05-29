"""
OCRCoderIQ — 基于 Qwen3-VL-2B-Instruct 的代码图谱补全模型。

组件:
  - ForwardModel: 封装 Qwen3-VL 模型的加载与推理
  - ContextDatabase: 记录多轮对话的上下文状态
  - OCRCoderIQ:  主控类，协调子图渲染、上下文压缩、模型推理
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from renderer import CodeGraphRenderer
from AgentOCR.agentocr.context_compressor import (
    ContextCompressor,
    Message,
)
from prompt_template import TASK_PROMPT


# ---------------------------------------------------------------------------
# 1. ForwardModel — 模型加载与推理封装
# ---------------------------------------------------------------------------

class ForwardModel:
    """封装 Qwen3-VL 模型的加载和推理。"""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 8000,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate(self, messages: List[Dict[str, Any]], verbose: bool = True) -> str:
        """
        将 multimodal messages 送入 Qwen3-VL 模型并返回解码文本。

        Qwen3-VL 要求图片通过 process_vision_info 单独提取并传入 processor，
        不能只靠 apply_chat_template 一步完成。
        """
        from qwen_vl_utils import process_vision_info

        # Step 1: 生成纯文本 chat template（tokenize=False）
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Step 2: 从 messages 中提取并加载图片
        image_inputs, video_inputs = process_vision_info(messages)

        if verbose:
            n_images = len(image_inputs) if image_inputs else 0
            n_text_chars = len(text) if text else 0
            print(f"  [ForwardModel] text={n_text_chars} chars, images={n_images}, "
                  f"first image size={image_inputs[0].size if image_inputs and image_inputs[0] else 'N/A'}")

        # Step 3: processor 同时处理文本和图片
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        if verbose:
            print(f"  [ForwardModel] input_ids shape={inputs.input_ids.shape}")

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result = output_text[0]

        if verbose:
            n_new_tokens = len(generated_ids_trimmed[0])
            preview = result[:300].replace("\n", "\\n")
            print(f"  [ForwardModel] generated {n_new_tokens} new tokens, "
                  f"output preview: {preview}...")

        return result


# ---------------------------------------------------------------------------
# 2. ContextDatabase — 上下文记录
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """单轮对话记录。"""
    turn: int
    seed_id: int
    user_query: str
    last_user_query: Optional[str]
    graph_image_path: str
    context_image_paths: List[str]
    model_response: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ContextDatabase:
    """
    记录多轮对话的上下文状态。

    每轮对话存储:
    - 轮次编号、seed 节点 id
    - 用户查询 / 上一轮查询
    - 生成的子图路径和上下文图片路径
    - 模型回复

    支持 JSON 持久化。
    """

    def __init__(self, session_id: str, db_path: Optional[str] = None):
        self.session_id = session_id
        self.db_path = db_path
        self.records: List[TurnRecord] = []

        # 自动从已有文件加载
        if db_path and os.path.isfile(db_path):
            self.load()

    # ---- 写入 ----

    def add_turn(
        self,
        turn: int,
        seed_id: int,
        user_query: str,
        last_user_query: Optional[str],
        graph_image_path: str,
        context_image_paths: List[str],
        model_response: Optional[str] = None,
    ) -> TurnRecord:
        """记录一轮对话，返回创建的记录。"""
        record = TurnRecord(
            turn=turn,
            seed_id=seed_id,
            user_query=user_query,
            last_user_query=last_user_query,
            graph_image_path=graph_image_path,
            context_image_paths=list(context_image_paths),
            model_response=model_response,
        )
        self.records.append(record)
        return record

    def update_response(self, turn: int, response: str) -> bool:
        """更新某一轮记录的模型回复。"""
        for r in self.records:
            if r.turn == turn:
                r.model_response = response
                return True
        return False

    # ---- 查询 ----

    def get_turn(self, turn: int) -> Optional[TurnRecord]:
        """按轮次号检索。"""
        for r in self.records:
            if r.turn == turn:
                return r
        return None

    def get_last_turn(self) -> Optional[TurnRecord]:
        """获取最近一轮的记录。"""
        return self.records[-1] if self.records else None

    def get_last_response(self) -> Optional[str]:
        """获取最近一轮的模型回复。"""
        last = self.get_last_turn()
        return last.model_response if last else None

    def get_context_messages(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """
        将最近的 N 轮对话组装为 Qwen3-VL 格式的 messages 列表，
        可用于下一轮推理时的上下文注入。
        """
        messages: List[Dict[str, Any]] = []
        for r in self.records[-last_n:]:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": r.user_query}],
            })
            if r.model_response:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": r.model_response}],
                })
        return messages

    def __len__(self) -> int:
        return len(self.records)

    # ---- 持久化 ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "records": [
                {
                    "turn": r.turn,
                    "seed_id": r.seed_id,
                    "user_query": r.user_query,
                    "last_user_query": r.last_user_query,
                    "graph_image_path": r.graph_image_path,
                    "context_image_paths": r.context_image_paths,
                    "model_response": r.model_response,
                    "timestamp": r.timestamp,
                }
                for r in self.records
            ],
        }

    def save(self, db_path: Optional[str] = None) -> str:
        """持久化到 JSON 文件。返回写入路径。"""
        path = db_path or self.db_path
        if path is None:
            path = f"context_db_{self.session_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        return path

    def load(self, db_path: Optional[str] = None) -> None:
        """从 JSON 文件加载记录。"""
        path = db_path or self.db_path
        if not path or not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.session_id = data.get("session_id", self.session_id)
        self.records = [
            TurnRecord(
                turn=r["turn"],
                seed_id=r["seed_id"],
                user_query=r["user_query"],
                last_user_query=r.get("last_user_query"),
                graph_image_path=r["graph_image_path"],
                context_image_paths=r.get("context_image_paths", []),
                model_response=r.get("model_response"),
                timestamp=r.get("timestamp", ""),
            )
            for r in data["records"]
        ]


# ---------------------------------------------------------------------------
# 3. OCRCoderIQ — 主控类
# ---------------------------------------------------------------------------

class OCRCoderIQ:
    """
    多轮代码图谱补全控制器。

    每轮对话:
    1. 接收 seed_id、用户查询和上下文
    2. 调用 CodeGraphRenderer 渲染子图
    3. 调用 ContextCompressor 压缩对话上下文为图片
    4. 组装 multimodal messages
    5. 调用 ForwardModel 生成补全结果
    6. 将结果存入 ContextDatabase
    """

    def __init__(
        self,
        context: Optional[List[Message]],
        window_size: int,
        file_path: str,
        graph_render: CodeGraphRenderer,
        context_render: ContextCompressor,
        session_id: str,
        model: Optional[ForwardModel] = None,
        database: Optional[ContextDatabase] = None,
    ):
        self.len = len(context) if context else 0
        self.turn = self.len
        self.window_size = window_size
        self.graph_image_cache = file_path
        self.context_image_cache = file_path
        self.session_id = session_id
        self.graph_render = graph_render
        self.context_render = context_render

        # 新增组件
        self.model = model
        self.database = database or ContextDatabase(session_id=session_id)

    def __len__(self):
        return self.len

    def session_reset(
        self,
        window_size: int,
        file_path: str,
        graph_render: CodeGraphRenderer,
        context_render: ContextCompressor,
        _seed_id: int,
        session_id: str,
    ):
        """重置会话状态。"""
        self.len = 0
        self.turn = 0
        self.window_size = window_size
        self.graph_image_cache = file_path
        self.context_image_cache = file_path
        self.session_id = session_id
        self.graph_render = graph_render
        self.context_render = context_render
        print("Session reset successfully")

    def rationale_decouple(self, _context: List[Message]) -> Dict[str, Any]:
        """
        从模型输出中解耦出推理过程和最终答案。

        当前为占位实现，后续可在此处解析 CoT / rationale。
        """
        return {"rationale": "", "answer": ""}

    # ------------------------------------------------------------------
    # 核心方法: 构建消息 + 调用模型 + 记录上下文
    # ------------------------------------------------------------------

    def generateVQ(
        self,
        seed_id: int,
        context: List[Message],
        user_query: str,
        last_user_query: Optional[str] = None,
    ) -> Optional[str]:
        """
        根据 seed_id 生成子图，构建多模态消息，调用模型完成补全。

        Parameters
        ----------
        seed_id : int
            中心节点 id
        context : list[Message]
            当前对话的完整上下文（AgentOCR Message 列表）
        user_query : str
            当前轮的用户查询
        last_user_query : str or None
            上一轮的用户查询（首轮为 None）

        Returns
        -------
        str or None
            模型生成的补全文本，如果模型不可用则返回 None
        """
        # --- 1. 渲染当前轮的子图 ---
        # dpi=100 → 1600×800 像素，VLM 可辨认图中文字；max_nodes=100 控制渲染速度
        new_graph_image = self.graph_render.render(
            seed_id,
            k=2,
            max_nodes=100,
            output_path=os.path.join(self.graph_image_cache, f"turn-{self.turn}"),
            show=False,
            figsize=(16, 8),
            dpi=100,
        )
        new_graph_image = os.path.abspath(new_graph_image)

        # --- 2. 压缩对话上下文为图片 ---
        new_context_images = self.context_render.compress_incremental(
            context, session_id=self.session_id
        )
        if len(new_context_images) >= self.window_size:
            new_context_images = new_context_images[: self.window_size]

        # 上下文图片落盘
        context_image_paths: List[str] = []
        for i, img in enumerate(new_context_images):
            path = os.path.join(self.context_image_cache, f"page_{self.turn}_{i+1}.png")
            img.save(path)
            context_image_paths.append(os.path.abspath(path))

        # --- 3. 组装 multimodal messages ---

        # system message
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": TASK_PROMPT}],
            }
        ]

        # 注入历史对话（前几轮）
        history_msgs = self.database.get_context_messages(last_n=3)
        messages.extend(history_msgs)

        # 当前轮 user message
        user_content: List[Dict[str, Any]] = []

        # 上一轮的图（如果存在）
        if self.turn >= 1 and last_user_query:
            last_graph_path = os.path.join(
                self.graph_image_cache, f"turn-{self.turn - 1}"
            )
            # render 输出可能带 .png 后缀
            for ext in (".png", ""):
                candidate = last_graph_path + ext
                if os.path.isfile(candidate):
                    user_content.append({"type": "image", "image": os.path.abspath(candidate)})
                    break
            user_content.append({
                "type": "text",
                "text": f"Last User Query: {last_user_query}",
            })

        # 当前轮的图
        user_content.append({"type": "image", "image": new_graph_image})
        user_content.append({
            "type": "text",
            "text": f"Current User Query: {user_query}",
        })

        # 上下文图片
        if context_image_paths:
            user_content.append({"type": "text", "text": "Context Images:"})
            for img_path in context_image_paths:
                user_content.append({"type": "image", "image": img_path})

        messages.append({"role": "user", "content": user_content})

        # --- 4. 记录到数据库 ---
        self.database.add_turn(
            turn=self.turn,
            seed_id=seed_id,
            user_query=user_query,
            last_user_query=last_user_query,
            graph_image_path=new_graph_image,
            context_image_paths=context_image_paths,
        )

        # --- 5. 调用模型 ---
        if self.model is not None:
            response = self.model.generate(messages)
            self.database.update_response(self.turn, response)
        else:
            response = None
            print("[Warn] No model provided — skipping inference.")

        self.turn += 1
        self.len += 1

        return response


# ---------------------------------------------------------------------------
# 4. CLI / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_path = os.path.abspath("./Qwen3-VL-2B-Instruct")

    # 初始化模型
    print("Loading model ...")
    model = ForwardModel(model_path=model_path)

    # Demo：不涉及具体图文件，仅验证模型推理链路
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant in image OCR field, "
                            "you need to parse all the context in each image.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.abspath(
                        os.path.join(
                            os.path.dirname(__file__),
                            "tmp/context_compressor_test/long_balanced_page1.png",
                        )
                    ),
                },
                {
                    "type": "image",
                    "image": os.path.abspath(
                        os.path.join(
                            os.path.dirname(__file__),
                            "tmp/context_compressor_test/long_balanced_page2.png",
                        )
                    ),
                },
            ],
        },
    ]

    response = model.generate(messages)
    print(response)
