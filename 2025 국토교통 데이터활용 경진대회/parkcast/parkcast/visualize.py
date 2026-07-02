"""검출 결과 시각화."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .inference import OccupancyResult
from .utils import is_empty_class


def _draw_instances(ax, result: OccupancyResult, linewidth: float = 1.5) -> None:
    """박스(detect) 또는 polygon(seg) 인스턴스를 ax에 그림.

    result.masks_xy가 있으면(YOLOv8n-seg 결과) 반투명 채움 polygon으로,
    없으면(기존 detect 결과) 윤곽선 박스로 그림 — 두 task를 같은 함수로 지원.
    """
    if result.has_masks:
        for poly, name in zip(result.masks_xy, result.class_names):
            color = "lime" if is_empty_class(name) else "red"
            ax.add_patch(
                mpatches.Polygon(
                    poly, closed=True, linewidth=linewidth,
                    edgecolor=color, facecolor=color, alpha=0.25,
                )
            )
    else:
        for (x1, y1, x2, y2), name in zip(result.boxes_xyxy, result.class_names):
            color = "lime" if is_empty_class(name) else "red"
            ax.add_patch(
                mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=linewidth, edgecolor=color, facecolor="none",
                )
            )


def draw_occupancy(
    image_path: str | Path,
    result: OccupancyResult,
    save_path: str | Path | None = None,
    show: bool = True,
    figsize=(12, 12),
) -> None:
    """단일 이미지에 박스 + 점유율 헤더를 그려 시각화/저장.

    초록 = empty, 빨강 = occupied.
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    _draw_instances(ax, result)

    title = (
        f"Empty: {result.n_empty}  |  Occupied: {result.n_occupied}  |  Total: {result.n_total}\n"
        f"Occupancy rate: {result.occupancy_pct:.1f}%"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_failure_grid(
    rows: List[dict],
    predictor,
    class_names: List[str],
    save_path: str | Path | None = None,
    n_rows: int = 2,
    n_cols: int = 3,
) -> None:
    """실패 케이스 K개를 그리드로 시각화.

    Args:
        rows: 각 dict는 'img_path', 'gt_total', 'pred_total', 'occ_rate_gt', 'occ_rate_pred' 보유
        predictor: OccupancyPredictor 인스턴스
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, row in zip(axes, rows):
        result = predictor.predict(row["img_path"], conf=0.4)
        img = cv2.imread(row["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        _draw_instances(ax, result, linewidth=1.0)

        ax.set_title(
            f"GT total={row['gt_total']} ({row['occ_rate_gt']*100:.0f}%)  |  "
            f"Pred total={row['pred_total']} ({row['occ_rate_pred']*100:.0f}%)",
            fontsize=10,
        )
        ax.axis("off")

    plt.suptitle("점유율 예측이 가장 어긋난 케이스 (high occupancy-rate error)", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def parse_yolo_seg_label(
    label_path: str | Path, img_w: int, img_h: int
) -> List[Tuple[int, np.ndarray]]:
    """YOLO-seg 라벨 txt(`class x1 y1 x2 y2 ...`, 0~1 정규화)를 픽셀 좌표 polygon으로 파싱."""
    instances = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            instances.append((cls, coords))
    return instances


def plot_yolo_seg_label_sample(
    image_path: str | Path,
    label_path: str | Path,
    class_names: List[str],
    save_path: str | Path | None = None,
    show: bool = True,
    figsize=(12, 12),
) -> None:
    """YOLO-seg 라벨 파일을 이미지 위에 직접 그려서 "각진 사각형이 아니라 진짜 윤곽인지" 검증.

    scripts/sam_auto_label.py로 만든 라벨을 학습 전에 눈으로 확인하는 용도.
    OccupancyPredictor 없이 라벨 파일만으로 그리므로 학습 전 데이터 검증에도 씀.
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    instances = parse_yolo_seg_label(label_path, W, H)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    for cls, pts in instances:
        name = class_names[cls] if cls < len(class_names) else str(cls)
        color = "lime" if is_empty_class(name) else "red"
        ax.add_patch(
            mpatches.Polygon(pts, closed=True, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.25)
        )
    ax.set_title(f"{Path(image_path).name} — {len(instances)} instances (YOLO-seg 라벨 검증)", fontsize=12)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_yolo_seg_grid(
    samples: List[Tuple[str, str]],
    class_names: List[str],
    save_path: str | Path | None = None,
    n_rows: int = 2,
    n_cols: int = 3,
) -> None:
    """여러 이미지의 YOLO-seg 라벨을 그리드로 검증 시각화 (samples: [(image_path, label_path), ...]).

    SAM 자동 라벨링(scripts/sam_auto_label.py) 산출물이 각진 사각형이 아니라 실제 차량/칸
    윤곽을 따라가는지 한 번에 여러 장 비교하는 용도 — README의 검증 그림에 그대로 씀.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, (img_path, lbl_path) in zip(axes, samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        instances = parse_yolo_seg_label(lbl_path, W, H)

        ax.imshow(img)
        for cls, pts in instances:
            name = class_names[cls] if cls < len(class_names) else str(cls)
            color = "lime" if is_empty_class(name) else "red"
            ax.add_patch(
                mpatches.Polygon(pts, closed=True, linewidth=1.0, edgecolor=color, facecolor=color, alpha=0.25)
            )
        ax.set_title(f"{Path(img_path).name} ({len(instances)} instances)", fontsize=9)
        ax.axis("off")

    plt.suptitle("SAM 자동 라벨링 검증 — 각진 사각형이 아니라 실제 윤곽인지 확인", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
