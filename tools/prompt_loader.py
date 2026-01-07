from __future__ import annotations

from pathlib import Path
import yaml

_PROMPTS_CACHE = None


def load_prompts() -> dict:
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "prompts.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            _PROMPTS_CACHE = yaml.safe_load(handle)
    return _PROMPTS_CACHE
