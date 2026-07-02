"""SAM(Segment Anything) box-prompted 자동 라벨링 — GT bbox로 진짜 픽셀 단위 마스크 획득.

`data.py::coco_to_yolo_seg`는 COCO의 `segmentation` 필드를 그대로 쓰는데, PKLot 원본은
이 필드가 있어도 bbox에서 파생된 사각형뿐이라 "진짜 윤곽"이 아님(각진 사각형 그대로).
실제 차량/빈칸 윤곽을 얻으려면 이미 검증된 YOLOv8n bbox(GT)를 SAM의 box prompt로 넣어
픽셀 단위 분할을 받아야 함. Ultralytics에 SAM이 통합돼 있어(`from ultralytics import SAM`)
추가 설치 없이 바로 씀.

파이프라인: COCO GT bbox 로드 → 이미지당 batch_size개씩 박스를 묶어 SAM에 프롬프트
→ mask(H,W) → cv2.findContours + approxPolyDP로 polygon 단순화 → 실패(컨투어 없음 등)
시 bbox 4꼭짓점으로 fallback(SamLabelingStats에 정직하게 카운트).

리소스 제약(PKLot 12,416장 × 평균 30~70박스, 무료 T4 기준)을 고려해 split별 서브셋
크기를 지정할 수 있음 — 포트폴리오 목적이면 대표 서브셋으로 충분함(README 참조).
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm.auto import tqdm

from .data import _link_or_copy, load_coco


def select_subset_image_ids(coco: Dict, n: Optional[int], seed: int = 42) -> List[int]:
    """전체 이미지 중 n장을 시드 고정 랜덤 샘플링. n이 None이면 전체 이미지."""
    all_ids = [im["id"] for im in coco["images"]]
    if n is None or n >= len(all_ids):
        return all_ids
    rng = random.Random(seed)
    return rng.sample(all_ids, n)


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.01) -> Optional[np.ndarray]:
    """binary mask(H,W) → cv2.findContours+approxPolyDP로 단순화한 polygon (M,2), 픽셀 좌표.

    가장 넓은 외곽 컨투어 하나만 사용(SAM box prompt는 박스당 마스크 1개이므로 보통 충분).
    컨투어가 없거나 면적이 0이면 None(호출부에서 bbox fallback 처리).
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_frac * peri, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2).astype(np.float32)


def _bbox_corners_xyxy(xyxy: List[float]) -> List[Tuple[float, float]]:
    x1, y1, x2, y2 = xyxy
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


@dataclass
class SamLabelingStats:
    n_images: int = 0
    n_boxes: int = 0
    n_sam_success: int = 0
    n_bbox_fallback: int = 0

    @property
    def fallback_rate(self) -> float:
        return self.n_bbox_fallback / max(self.n_boxes, 1)


def sam_label_split(
    src_dir: str | Path,
    dst_dir: str | Path,
    cat_id_to_yolo: Dict[int, int],
    sam: "object",
    batch_size: int = 50,
    subset_n: Optional[int] = None,
    epsilon_frac: float = 0.01,
    seed: int = 42,
    device: Optional[str] = None,
    use_symlink: bool = True,
) -> SamLabelingStats:
    """한 split(train/valid/test) 전체를 SAM box-prompted 라벨링해서 YOLO-seg 라벨로 저장.

    sam은 이미 로드된 `ultralytics.SAM` 인스턴스(split마다 새로 로드하지 않도록 호출부에서 공유).
    """
    src_dir = Path(src_dir)
    dst_img = Path(dst_dir) / "images"
    dst_lbl = Path(dst_dir) / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    coco = load_coco(src_dir / "_annotations.coco.json")
    img_to_anns: Dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        img_to_anns[a["image_id"]].append(a)
    img_id_to_info = {im["id"]: im for im in coco["images"]}

    keep_ids = select_subset_image_ids(coco, subset_n, seed=seed)
    stats = SamLabelingStats()

    for img_id in tqdm(sorted(keep_ids), desc=f"SAM labeling ({src_dir.name})"):
        info = img_id_to_info[img_id]
        fname = info["file_name"]
        W, H = info["width"], info["height"]
        anns = img_to_anns.get(img_id, [])
        if not anns:
            continue

        src_path = src_dir / fname
        dst_path = dst_img / fname
        if not dst_path.exists():
            _link_or_copy(src_path, dst_path, use_symlink)

        lines: List[str] = []
        for batch_start in range(0, len(anns), batch_size):
            batch = anns[batch_start:batch_start + batch_size]
            xyxy_boxes = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in batch)]

            predict_kwargs = {"verbose": False}
            if device:
                predict_kwargs["device"] = device
            result = sam(str(src_path), bboxes=xyxy_boxes, **predict_kwargs)[0]
            masks = (
                result.masks.data.cpu().numpy() if result.masks is not None else np.empty((0,))
            )

            for i, ann in enumerate(batch):
                cls = cat_id_to_yolo[ann["category_id"]]
                stats.n_boxes += 1

                poly_px = mask_to_polygon(masks[i], epsilon_frac=epsilon_frac) if i < len(masks) else None
                if poly_px is None:
                    poly_px = _bbox_corners_xyxy(xyxy_boxes[i])
                    stats.n_bbox_fallback += 1
                else:
                    stats.n_sam_success += 1

                coords = " ".join(f"{px / W:.6f} {py / H:.6f}" for px, py in poly_px)
                lines.append(f"{cls} {coords}")

        lbl_path = dst_lbl / (Path(fname).stem + ".txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))
        stats.n_images += 1

    return stats


def sam_label_dataset(
    raw_root: str | Path,
    yolo_root: str | Path,
    splits: Tuple[str, ...] = ("train", "valid", "test"),
    subset_n: Optional[Dict[str, int]] = None,
    sam_model: str = "sam_b.pt",
    batch_size: int = 50,
    epsilon_frac: float = 0.01,
    seed: int = 42,
    device: Optional[str] = None,
    use_symlink: bool = True,
) -> Dict[str, SamLabelingStats]:
    """전체 split을 SAM 자동 라벨링해서 YOLO-seg 데이터셋(images+labels+data.yaml)을 만듦.

    subset_n={'train': 2000, 'valid': 300, 'test': 300}처럼 split별 서브셋 크기를 지정 가능
    (리소스 제약 대응 — None이면 전체 사용). sam_model은 ultralytics가 자동 다운로드함
    (mobile_sam.pt가 가장 가벼움, sam_b.pt/sam2_b.pt는 더 무겁지만 정교함).
    """
    from ultralytics import SAM

    raw_root = Path(raw_root)
    yolo_root = Path(yolo_root)
    subset_n = subset_n or {}

    first_coco = load_coco(raw_root / splits[0] / "_annotations.coco.json")
    cats_sorted = sorted(first_coco["categories"], key=lambda x: x["id"])
    cat_id_to_yolo = {cat["id"]: i for i, cat in enumerate(cats_sorted)}
    yolo_names = [cat["name"] for cat in cats_sorted]

    sam = SAM(sam_model)

    all_stats: Dict[str, SamLabelingStats] = {}
    for split in splits:
        stats = sam_label_split(
            raw_root / split, yolo_root / split, cat_id_to_yolo, sam,
            batch_size=batch_size, subset_n=subset_n.get(split), epsilon_frac=epsilon_frac,
            seed=seed, device=device, use_symlink=use_symlink,
        )
        all_stats[split] = stats
        print(
            f"  {split}: {stats.n_images} images, {stats.n_boxes} boxes, "
            f"SAM success {stats.n_sam_success}, bbox fallback {stats.n_bbox_fallback} "
            f"({stats.fallback_rate * 100:.1f}%)"
        )

    yaml_content = (
        f"path: {yolo_root}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(yolo_names)}\nnames: {yolo_names}\n"
    )
    with open(yolo_root / "data.yaml", "w") as f:
        f.write(yaml_content)

    return all_stats
