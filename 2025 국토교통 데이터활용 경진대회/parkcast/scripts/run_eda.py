"""EDA 시각화 (split 분포 + 샘플 박스 그림) 실행 + 저장.

사용:
    python scripts/run_eda.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parkcast.data import split_stats
from parkcast.eda import plot_sample_boxes, plot_split_distribution
from parkcast.utils import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.paths
    ensure_dirs(paths.results_dir)

    out = Path(paths.results_dir)

    print(f"[1/2] Split distribution → {out}/eda_distribution.png")
    stats = split_stats(paths.raw_root)
    plot_split_distribution(stats, save_path=out / "eda_distribution.png")

    print(f"[2/2] Sample boxes → {out}/eda_samples.png")
    plot_sample_boxes(
        paths.raw_root, split="train", n_samples=args.n_samples,
        save_path=out / "eda_samples.png",
    )

    print("Done.")


if __name__ == "__main__":
    main()
