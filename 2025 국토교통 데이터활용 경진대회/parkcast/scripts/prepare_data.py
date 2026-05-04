"""COCO → YOLO format 변환.

사용:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 패키지 root를 sys.path에 추가 (scripts/에서 직접 실행 시)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parkcast.data import coco_to_yolo, split_stats
from parkcast.utils import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.paths

    print(f"[1/3] Split 통계")
    stats = split_stats(paths.raw_root)
    print(stats.to_string(index=False))

    print(f"\n[2/3] COCO → YOLO 변환  ({paths.raw_root} → {paths.yolo_root})")
    ensure_dirs(paths.yolo_root)
    yolo_names, cat_id_to_yolo = coco_to_yolo(paths.raw_root, paths.yolo_root)

    print(f"\n[3/3] 완료")
    print(f"  YOLO classes: {yolo_names}")
    print(f"  cat_id → yolo_idx: {cat_id_to_yolo}")
    print(f"  data.yaml: {paths.yolo_root}/data.yaml")


if __name__ == "__main__":
    main()
