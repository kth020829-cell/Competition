"""학습된 모델 test set 평가 + 점유율 비교 + 실패 케이스 시각화.

사용:
    python scripts/evaluate.py --weights models/yolov8n_pklot_v1_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parkcast.evaluate import compare_gt_vs_pred, evaluate_on_test, top_failure_cases
from parkcast.inference import OccupancyPredictor
from parkcast.utils import YOLO_CLASS_NAMES, ensure_dirs, load_config
from parkcast.visualize import plot_failure_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", required=True, help="best.pt 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.paths
    eval_cfg = cfg.evaluate.raw

    ensure_dirs(paths.results_dir)
    out = Path(paths.results_dir)
    data_yaml = Path(paths.yolo_root) / "data.yaml"

    # ── 1) Test mAP/precision/recall
    print("[1/3] Test set 평가")
    metrics = evaluate_on_test(
        weights=args.weights,
        data_yaml=data_yaml,
        project_dir=paths.results_dir,
        run_name=f"{cfg.train.run_name}_test",
        imgsz=cfg.train.imgsz,
        batch=cfg.train.batch,
        class_names=YOLO_CLASS_NAMES,
    )
    print(f"  mAP50:     {metrics.mAP50:.4f}")
    print(f"  mAP50-95:  {metrics.mAP50_95:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    for k, v in metrics.per_class_mAP50.items():
        print(f"    {k:20s} mAP50={v:.4f}")

    # ── 2) GT vs prediction 점유율 비교
    print(f"\n[2/3] Occupancy comparison ({eval_cfg['failure_sample_n']} samples)")
    predictor = OccupancyPredictor(args.weights, class_names=YOLO_CLASS_NAMES)
    diff_df = compare_gt_vs_pred(
        predictor=predictor,
        raw_test_dir=Path(paths.raw_root) / "test",
        yolo_test_dir=Path(paths.yolo_root) / "test",
        sample_n=eval_cfg["failure_sample_n"],
        conf=eval_cfg["conf"],
    )
    diff_df.to_csv(out / "gt_vs_pred.csv", index=False)
    print(f"  평균 box count diff:  {diff_df['count_diff'].mean():.2f}")
    print(f"  평균 occupancy diff:  {diff_df['occ_rate_diff'].mean()*100:.2f}%")

    # ── 3) Top-K 실패 케이스 시각화
    print(f"\n[3/3] Top-{eval_cfg['failure_topk']} failure cases")
    worst = top_failure_cases(diff_df, k=eval_cfg["failure_topk"])
    plot_failure_grid(
        rows=worst,
        predictor=predictor,
        class_names=YOLO_CLASS_NAMES,
        save_path=out / "failure_cases.png",
    )

    # ── 요약 파일
    summary = (
        "ParkCast Vision — Evaluation Summary\n"
        "=====================================\n"
        f"weights: {args.weights}\n"
        f"  mAP50:     {metrics.mAP50:.4f}\n"
        f"  mAP50-95:  {metrics.mAP50_95:.4f}\n"
        f"  Precision: {metrics.precision:.4f}\n"
        f"  Recall:    {metrics.recall:.4f}\n"
        f"\nOccupancy estimation (n={len(diff_df)}):\n"
        f"  mean box count diff:  {diff_df['count_diff'].mean():.2f}\n"
        f"  mean occupancy diff:  {diff_df['occ_rate_diff'].mean()*100:.2f}%\n"
    )
    with open(out / "evaluation_summary.txt", "w") as f:
        f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
