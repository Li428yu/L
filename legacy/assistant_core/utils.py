from __future__ import annotations

import math


def estimate_cost_hint(chunk_count: int, average_chars_per_chunk: int = 900) -> str:
    estimated_tokens = math.ceil(chunk_count * average_chars_per_chunk / 4)
    return f"本次建立索引预计会处理约 {estimated_tokens} 个文本 token。"
