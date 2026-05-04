"""Test set 평가 + GT vs prediction 점유율 비교 + 실패 케이스 추출."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from .data import build_image_to_gt
from .inference import OccupancyPredictor


@dataclass
class EvalMetrics:
    """YOLO val() 결과를 정리."""
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    per_class_mAP50: Dict[str, float]


def evaluate_on_test(
    weights: str | Path,
    data_yaml: str | Path,
    project_dir: str | Path,
    run_name: str = "yolov8n_pklot_v1_test",
    imgsz: int = 640,
    batch: int = 32,
    class_names: Optional[List[str]] = None,
) -> EvalMetrics:
    """Ultralytics val()을 호출하고 metric을 EvalMetrics dataclass로 반환."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        project=str(project_dir),
        name=run_name,
        plots=True,
    )

    per_class = {}
    if class_names is not None:
        maps = metrics.box.maps
        for i, name in enumerate(class_names):
            if i < len(maps):
                per_class[name] = float(maps[i])

    return EvalMetrics(
        mAP50=float(metrics.box.map50),
        mAP50_95=float(metrics.box.map),
        precision=float(metrics.box.mp),
        recall=float(metrics.box.mr),
        per_class_mAP50=per_class,
    )


def compare_gt_vs_pred(
    predictor: OccupancyPredictor,
    raw_test_dir: str | Path,
    yolo_test_dir: str | Path,
    sample_n: int = 200,
    conf: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """테스트 이미지 N장에 대해 GT 점유 카운트 vs 모델 예측 카운트 비교.

    Returns:
        각 행이 한 이미지: gt_*, pred_*, count_diff, occ_rate_diff 포함
    """
    raw_test_dir = Path(raw_test_dir)
    yolo_test_dir = Path(yolo_test_dir)

    img_to_gt, img_id_to_fname = build_image_to_gt(
        raw_test_dir / "_annotations.coco.json"
    )

    rng = random.Random(seed)
    sample_ids = rng.sample(
        list(img_id_to_fname.keys()), min(sample_n, len(img_id_to_fname))
    )

    rows = []
    for img_id in tqdm(sample_ids, desc="GT vs pred"):
        fname = img_id_to_fname[img_id]
        img_path = yolo_test_dir / "images" / fname
        if not img_path.exists():
            continue

        result = predictor.predict(img_path, conf=conf)
        gt = img_to_gt.get(img_id, {"empty": 0, "occupied": 0})
        gt_total = gt["empty"] + gt["occupied"]
        pred_total = result.n_total

        rows.append({
            "fname": fname,
            "img_path": str(img_path),
            "gt_empty": gt["empty"],
            "gt_occupied": gt["occupied"],
            "gt_total": gt_total,
            "pred_empty": result.n_empty,
            "pred_occupied": result.n_occupied,
            "pred_total": pred_total,
            "count_diff": abs(gt_total - pred_total),
            "occ_rate_gt": gt["occupied"] / max(gt_total, 1),
            "occ_rate_pred": result.n_occupied / max(pred_total, 1),
        })

    df = pd.DataFrame(rows)
    df["occ_rate_diff"] = (df["occ_rate_gt"] - df["occ_rate_pred"]).abs()
    return df


def top_failure_cases(diff_df: pd.DataFrame, k: int = 6) -> List[Dict]:
    """점유율 오차가 가장 큰 K개의 케이스를 dict 리스트로 반환."""
    return diff_df.nlargest(k, "occ_rate_diff").to_dict(orient="records")
