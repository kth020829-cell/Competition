"""데이터 로딩 및 COCO → YOLO format 변환.

PKLot dataset (Roboflow v2)는 COCO format으로 배포되지만 Ultralytics YOLO는
YOLO format(class cx cy w h, normalized)을 요구하므로 변환이 필요.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm.auto import tqdm


SPLITS = ("train", "valid", "test")


def load_coco(coco_path: str | Path) -> Dict:
    with open(coco_path, "r") as f:
        return json.load(f)


def split_stats(raw_root: str | Path) -> pd.DataFrame:
    """각 split의 이미지/어노테이션/클래스별 카운트를 DataFrame으로 반환."""
    rows = []
    for split in SPLITS:
        coco_path = Path(raw_root) / split / "_annotations.coco.json"
        if not coco_path.exists():
            continue
        c = load_coco(coco_path)
        cat_map = {x["id"]: x["name"] for x in c["categories"]}
        label_counts = Counter(cat_map[a["category_id"]] for a in c["annotations"])
        n_img = max(len(c["images"]), 1)
        rows.append({
            "split": split,
            "n_images": len(c["images"]),
            "n_annotations": len(c["annotations"]),
            "avg_boxes_per_img": round(len(c["annotations"]) / n_img, 1),
            **{k: v for k, v in label_counts.items()},
        })
    return pd.DataFrame(rows)


def coco_to_yolo(
    raw_root: str | Path,
    yolo_root: str | Path,
    splits: Tuple[str, ...] = SPLITS,
    use_symlink: bool = True,
) -> Tuple[List[str], Dict[int, int]]:
    """COCO format을 YOLO format으로 변환.

    출력 구조:
        yolo_root/
          ├── train/{images,labels}/
          ├── valid/{images,labels}/
          ├── test/{images,labels}/
          └── data.yaml

    Returns:
        yolo_names: 정렬된 YOLO 클래스 이름 리스트
        cat_id_to_yolo: COCO category_id → YOLO 0-base 인덱스 매핑
    """
    raw_root = Path(raw_root)
    yolo_root = Path(yolo_root)

    # 첫 split에서 카테고리 정렬해서 인덱스 결정 (전 split이 같은 카테고리 가짐)
    first_coco = load_coco(raw_root / splits[0] / "_annotations.coco.json")
    cats_sorted = sorted(first_coco["categories"], key=lambda x: x["id"])
    cat_id_to_yolo = {cat["id"]: i for i, cat in enumerate(cats_sorted)}
    yolo_names = [cat["name"] for cat in cats_sorted]

    for split in splits:
        _convert_split(raw_root / split, yolo_root / split, cat_id_to_yolo, use_symlink)

    # data.yaml 작성
    yaml_content = (
        f"path: {yolo_root}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: {len(yolo_names)}\n"
        f"names: {yolo_names}\n"
    )
    with open(yolo_root / "data.yaml", "w") as f:
        f.write(yaml_content)

    return yolo_names, cat_id_to_yolo


def _convert_split(
    src_dir: Path,
    dst_root: Path,
    cat_id_to_yolo: Dict[int, int],
    use_symlink: bool,
) -> None:
    dst_img = dst_root / "images"
    dst_lbl = dst_root / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    coco = load_coco(src_dir / "_annotations.coco.json")
    img_to_anns: Dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        img_to_anns[a["image_id"]].append(a)

    for img_info in tqdm(coco["images"], desc=f"{src_dir.name} convert"):
        fname = img_info["file_name"]
        W, H = img_info["width"], img_info["height"]

        # 이미지 옮기기
        src_path = src_dir / fname
        dst_path = dst_img / fname
        if not dst_path.exists():
            if use_symlink:
                try:
                    os.symlink(src_path.resolve(), dst_path)
                except (OSError, FileExistsError):
                    # Colab/Windows 등 symlink 불가 환경
                    import shutil
                    shutil.copy(src_path, dst_path)
            else:
                import shutil
                shutil.copy(src_path, dst_path)

        # 라벨 파일 작성
        lines = []
        for a in img_to_anns[img_info["id"]]:
            cls = cat_id_to_yolo[a["category_id"]]
            x, y, w, h = a["bbox"]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw, nh = w / W, h / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        lbl_path = dst_lbl / (Path(fname).stem + ".txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))


def build_image_to_gt(coco_path: str | Path) -> Tuple[Dict[int, Dict[str, int]], Dict[int, str]]:
    """COCO json에서 image_id → {empty: n, occupied: n} 매핑 + image_id → filename 매핑."""
    coco = load_coco(coco_path)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_to_gt: Dict[int, Dict[str, int]] = defaultdict(lambda: {"empty": 0, "occupied": 0})
    for a in coco["annotations"]:
        name = cat_map[a["category_id"]].lower()
        if "empty" in name:
            img_to_gt[a["image_id"]]["empty"] += 1
        elif "occupied" in name:
            img_to_gt[a["image_id"]]["occupied"] += 1
    img_id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}
    return dict(img_to_gt), img_id_to_fname
