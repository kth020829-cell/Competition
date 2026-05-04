"""YOLOv8 학습 실행.

사용:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parkcast.train import train_yolo
from parkcast.utils import ensure_dirs, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    paths = cfg.paths
    train_cfg = cfg.train.raw

    ensure_dirs(paths.model_dir, paths.results_dir)

    data_yaml = Path(paths.yolo_root) / "data.yaml"
    assert data_yaml.exists(), (
        f"data.yaml not found at {data_yaml}. "
        f"Run `python scripts/prepare_data.py` first."
    )

    print(f"학습 시작: {train_cfg['model']}, {train_cfg['epochs']} epochs, batch={train_cfg['batch']}")
    best = train_yolo(data_yaml, paths.results_dir, train_cfg)

    # 드라이브에 영속 저장
    target = Path(paths.model_dir) / f"{train_cfg['run_name']}_best.pt"
    shutil.copy(best, target)
    print(f"\nBest weights: {best}")
    print(f"Saved to:     {target}")


if __name__ == "__main__":
    main()
