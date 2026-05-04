"""단일 이미지 점유율 예측.

사용:
    python scripts/predict.py --weights models/best.pt --image lot.jpg
    python scripts/predict.py --weights models/best.pt --image lot.jpg --save out.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parkcast.inference import OccupancyPredictor
from parkcast.utils import YOLO_CLASS_NAMES
from parkcast.visualize import draw_occupancy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--save", help="시각화 저장 경로 (생략 시 화면 출력)")
    args = parser.parse_args()

    pred = OccupancyPredictor(args.weights, class_names=YOLO_CLASS_NAMES)
    result = pred.predict(args.image, conf=args.conf)

    print(f"Image:           {args.image}")
    print(f"  Empty:         {result.n_empty}")
    print(f"  Occupied:      {result.n_occupied}")
    print(f"  Total:         {result.n_total}")
    print(f"  Occupancy:     {result.occupancy_pct:.1f}%")

    draw_occupancy(args.image, result, save_path=args.save, show=args.save is None)


if __name__ == "__main__":
    main()
