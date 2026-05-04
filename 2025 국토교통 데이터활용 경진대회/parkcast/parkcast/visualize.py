"""검출 결과 시각화."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .inference import OccupancyResult
from .utils import is_empty_class


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

    for (x1, y1, x2, y2), name in zip(result.boxes_xyxy, result.class_names):
        color = "lime" if is_empty_class(name) else "red"
        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor="none"
            )
        )

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

        for (x1, y1, x2, y2), name in zip(result.boxes_xyxy, result.class_names):
            color = "lime" if is_empty_class(name) else "red"
            ax.add_patch(
                mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=1.0, edgecolor=color, facecolor="none"
                )
            )

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
