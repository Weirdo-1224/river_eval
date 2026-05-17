"""Local Qwen model adapter skeleton.

This is intentionally not registered yet. It marks the intended extension point
for future local multimodal Qwen experiments with memory modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.models.base import BaseModel


class QwenLocalModel(BaseModel):
    def __init__(self, model_path: str | Path, **_: Any) -> None:
        self.model_path = Path(model_path)

    def generate(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[Path],
        memory: Any = None,
        sample_id: str = "",
    ) -> str:
        raise NotImplementedError("Local Qwen inference is not implemented in this phase.")
