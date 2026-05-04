"""YOLOv8 학습 wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def train_yolo(
    data_yaml: str | Path,
    project_dir: str | Path,
    train_cfg: Dict[str, Any],
) -> Path:
    """YOLOv8 학습 후 best.pt 경로를 반환.

    Args:
        data_yaml: YOLO data.yaml 경로
        project_dir: Ultralytics project 폴더 (run 결과가 여기 저장됨)
        train_cfg: configs/default.yaml의 train 섹션 dict

    Returns:
        best weights 파일 경로
    """
    from ultralytics import YOLO

    model = YOLO(train_cfg["model"])

    model.train(
        data=str(data_yaml),
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        project=str(project_dir),
        name=train_cfg["run_name"],
        patience=train_cfg["patience"],
        cos_lr=train_cfg["cos_lr"],
        lr0=train_cfg["lr0"],
        degrees=train_cfg["degrees"],
        translate=train_cfg["translate"],
        scale=train_cfg["scale"],
        fliplr=train_cfg["fliplr"],
        mosaic=train_cfg["mosaic"],
        plots=True,
    )

    best_path = Path(project_dir) / train_cfg["run_name"] / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt not found at {best_path}")
    return best_path
