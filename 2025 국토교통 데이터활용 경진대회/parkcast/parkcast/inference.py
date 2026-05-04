"""학습된 YOLO 모델로 단일 이미지의 점유율 추정."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .utils import YOLO_CLASS_NAMES, is_empty_class, is_occupied_class


@dataclass
class OccupancyResult:
    """단일 이미지 점유율 추정 결과."""
    n_empty: int
    n_occupied: int
    n_total: int
    occupancy_rate: float                          # 0~1
    boxes_xyxy: np.ndarray                         # (N, 4)
    class_ids: np.ndarray                          # (N,)
    confidences: np.ndarray                        # (N,)
    class_names: List[str]                         # 박스별 클래스 이름

    @property
    def occupancy_pct(self) -> float:
        return self.occupancy_rate * 100


class OccupancyPredictor:
    """학습된 YOLOv8 모델을 wrap한 점유율 예측기.

    사용:
        pred = OccupancyPredictor("models/best.pt")
        result = pred.predict("parking_lot.jpg", conf=0.4)
        print(f"점유율: {result.occupancy_pct:.1f}%")
    """

    def __init__(
        self,
        weights: str | Path,
        class_names: Optional[List[str]] = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.class_names = class_names or YOLO_CLASS_NAMES

    def predict(self, image_path: str | Path, conf: float = 0.4) -> OccupancyResult:
        results = self.model(str(image_path), conf=conf, verbose=False)
        r = results[0]

        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        names = [self.class_names[c] for c in cls_ids]

        n_empty = sum(1 for n in names if is_empty_class(n))
        n_occupied = sum(1 for n in names if is_occupied_class(n))
        n_total = n_empty + n_occupied
        rate = (n_occupied / n_total) if n_total > 0 else 0.0

        return OccupancyResult(
            n_empty=n_empty,
            n_occupied=n_occupied,
            n_total=n_total,
            occupancy_rate=rate,
            boxes_xyxy=boxes,
            class_ids=cls_ids,
            confidences=confs,
            class_names=names,
        )
