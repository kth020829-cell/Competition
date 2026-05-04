"""EDA 시각화: 분포, 샘플 박스 그리기."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from .data import load_coco
from .utils import is_empty_class


def plot_split_distribution(stats_df: pd.DataFrame, save_path: str | Path | None = None) -> None:
    """split별 이미지/어노테이션 카운트 막대그래프."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    stats_df.set_index("split")[["n_images", "n_annotations"]].plot(
        kind="bar", ax=axes[0], color=["#264653", "#2a9d8f"]
    )
    axes[0].set_title("Images & annotations per split")
    axes[0].tick_params(axis="x", rotation=0)

    label_cols = [c for c in stats_df.columns if c.startswith("space-")]
    if label_cols:
        stats_df.set_index("split")[label_cols].plot(
            kind="bar", stacked=True, ax=axes[1], color=["#2a9d8f", "#e76f51"]
        )
        axes[1].set_title("Class distribution per split")
        axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_sample_boxes(
    raw_root: str | Path,
    split: str = "train",
    n_samples: int = 4,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> None:
    """COCO 어노테이션을 박스로 그려 샘플 시각화.

    초록 = empty, 빨강 = occupied.
    """
    coco = load_coco(Path(raw_root) / split / "_annotations.coco.json")
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_to_anns: Dict[int, List] = defaultdict(list)
    for a in coco["annotations"]:
        img_to_anns[a["image_id"]].append(a)

    rng = random.Random(seed)
    samples = rng.sample(coco["images"], min(n_samples, len(coco["images"])))

    n_cols = 2
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    axes = axes.flat if n_rows > 1 else [axes] if n_cols == 1 else axes

    for ax, img_info in zip(axes, samples):
        img = cv2.imread(str(Path(raw_root) / split / img_info["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        n_e, n_o = 0, 0
        for ann in img_to_anns[img_info["id"]]:
            x, y, w, h = ann["bbox"]
            cat_name = cat_map[ann["category_id"]]
            color = "lime" if is_empty_class(cat_name) else "red"
            if is_empty_class(cat_name):
                n_e += 1
            else:
                n_o += 1
            ax.add_patch(mpatches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor=color, facecolor="none"))

        ax.set_title(f"{img_info['file_name'][:30]}...\nEmpty={n_e}, Occupied={n_o}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
