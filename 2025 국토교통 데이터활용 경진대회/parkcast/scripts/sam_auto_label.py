"""SAM(Segment Anything) box-prompted 자동 라벨링 — 진짜 픽셀 단위 마스크로 YOLO-seg 데이터셋 생성.

이미 검증된 YOLOv8n GT bbox(mAP50 0.9944 학습에 쓰인 것과 동일한 COCO 어노테이션)를
SAM의 box prompt로 넣어 픽셀 단위 분할을 받고, cv2.findContours+approxPolyDP로 정리된
polygon으로 변환해 YOLO-seg 라벨을 만듦(parkcast/sam_label.py). scripts/prepare_data.py
--config configs/segment.yaml(COCO segmentation 필드 그대로/bbox fallback)보다 훨씬
정교한 "진짜 윤곽" 마스크를 얻지만 GPU가 필요함.

사용:
    python scripts/sam_auto_label.py --config configs/segment.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parkcast.sam_label import sam_label_dataset
from parkcast.utils import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segment.yaml")
    parser.add_argument("--device", default=None, help="예: cuda:0, cpu (생략하면 SAM 기본값)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.paths
    sam_cfg = cfg.sam_labeling.raw

    ensure_dirs(paths.yolo_root)

    print(f"[1/1] SAM box-prompted 라벨링 (model={sam_cfg['sam_model']}, "
          f"subset={sam_cfg.get('subset')})")
    stats = sam_label_dataset(
        raw_root=paths.raw_root,
        yolo_root=paths.yolo_root,
        subset_n=sam_cfg.get("subset"),
        sam_model=sam_cfg["sam_model"],
        batch_size=sam_cfg["batch_size"],
        epsilon_frac=sam_cfg["epsilon_frac"],
        seed=sam_cfg["seed"],
        device=args.device,
    )

    total_boxes = sum(s.n_boxes for s in stats.values())
    total_fallback = sum(s.n_bbox_fallback for s in stats.values())
    print(f"\n완료: {paths.yolo_root}/data.yaml")
    print(f"  전체 박스 {total_boxes:,}개 중 bbox fallback {total_fallback:,}개 "
          f"({total_fallback / max(total_boxes, 1) * 100:.1f}%)")
    print("  다음: python scripts/train.py --config configs/segment.yaml")


if __name__ == "__main__":
    main()
