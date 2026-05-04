"""공통 유틸리티: config 로딩, 경로 관리, 시드."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


@dataclass
class Config:
    """YAML config을 점-접근 가능한 dataclass로 wrap."""
    raw: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, key: str) -> Any:
        if key == "raw":
            raise AttributeError(key)
        if key in self.raw:
            v = self.raw[key]
            if isinstance(v, dict):
                return Config(raw=v)
            return v
        raise AttributeError(f"Config has no attribute '{key}'")

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.raw


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        return Config(raw=yaml.safe_load(f))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# PKLot YOLO class names (Roboflow PKLot v2 기준)
# 0: dummy supercategory, 1: empty, 2: occupied
YOLO_CLASS_NAMES = ["spaces", "space-empty", "space-occupied"]


def is_empty_class(class_name: str) -> bool:
    return "empty" in class_name.lower()


def is_occupied_class(class_name: str) -> bool:
    return "occupied" in class_name.lower()
