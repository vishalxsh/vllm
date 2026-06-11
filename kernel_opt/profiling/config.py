from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ProfileConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    warmup_tokens: int = 20
    profile_tokens: int = 10
    top_n: int = 20
    prompt: str = "Hello, how are you?"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
