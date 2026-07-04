"""Cross-lot(도메인 갭) 평가 — Random vs Date vs Lot split 한 번에 비교.

parkcast/domain.py 모듈을 사용해 이 로직을 CLI 한 번으로 재현함(노트북 버전은
notebooks/ParkCast_Week2_CrossLot.ipynb). Random split(이미 scripts/evaluate.py로
구한 결과)을 인자로 넘기면, Date split과 Lot(cross-domain) split을 새로 만들어
각각 학습·평가하고 셋을 비교한 막대그래프/CSV를 저장함.

사용 (prepare_data.py로 만든 data.yaml이 이미 있어야 함):
    python scripts/cross_lot_eval.py --config configs/default.yaml \
        --random-map50 0.9944 --random-map50-95 0.9886 \
        --random-precision 0.9977 --random-recall 0.9975
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parkcast.domain import (
    assign_date_split,
    assign_lot_split,
    build_image_metadata,
    build_yolo_split,
    compare_splits,
    discover_lots,
    extract_resnet50_embeddings,
    plot_split_comparison,
)
from parkcast.evaluate import EvalMetrics, evaluate_on_test
from parkcast.train import train_yolo
from parkcast.utils import YOLO_CLASS_NAMES, ensure_dirs, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-map50", type=float, required=True, help="기존 random-split 모델의 test mAP50")
    parser.add_argument("--random-map50-95", type=float, required=True)
    parser.add_argument("--random-precision", type=float, required=True)
    parser.add_argument("--random-recall", type=float, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    paths = cfg.paths
    train_cfg = cfg.train.raw

    data_yaml = Path(paths.yolo_root) / "data.yaml"
    assert data_yaml.exists(), (
        f"data.yaml not found at {data_yaml}. Run `python scripts/prepare_data.py` first."
    )

    out_dir = Path(paths.results_dir) / "cross_lot"
    ensure_dirs(out_dir)

    print("[1/5] 이미지 메타데이터 수집 + 파일명에서 날짜 파싱")
    meta = build_image_metadata(paths.yolo_root)
    n_dated = meta["date"].notna().sum()
    print(f"  총 {len(meta):,}장, 날짜 파싱 성공 {n_dated:,}장")

    print("\n[2/5] ResNet50 임베딩 추출 + K-Means로 주차장(lot) 자동 발견")
    embeddings = extract_resnet50_embeddings(meta["path"].tolist())
    cluster = discover_lots(embeddings, with_umap=False)
    meta["lot_cluster"] = cluster.labels
    print(f"  best k={cluster.best_k} (silhouette={cluster.silhouette_scores[cluster.best_k]:.4f})")
    print(meta["lot_cluster"].value_counts().to_string())

    print("\n[3/5] Date split / Lot split 구성 + YOLO 폴더 생성")
    meta["split_date"] = assign_date_split(meta)
    meta["split_lot"] = assign_lot_split(meta, cluster_col="lot_cluster")
    meta.to_csv(out_dir / "meta_with_splits.csv", index=False)

    date_yaml = build_yolo_split(meta, "split_date", paths.yolo_root, out_dir / "yolo_date", YOLO_CLASS_NAMES)
    lot_yaml = build_yolo_split(meta, "split_lot", paths.yolo_root, out_dir / "yolo_lot", YOLO_CLASS_NAMES)

    print("\n[4/5] Date split 학습+평가")
    date_best = train_yolo(date_yaml, out_dir, {**train_cfg, "run_name": "split_date"})
    date_metrics = evaluate_on_test(
        date_best, date_yaml, out_dir, run_name="split_date_test",
        imgsz=train_cfg["imgsz"], batch=train_cfg["batch"], class_names=YOLO_CLASS_NAMES,
    )

    print("\n      Lot split(cross-domain) 학습+평가")
    lot_best = train_yolo(lot_yaml, out_dir, {**train_cfg, "run_name": "split_lot"})
    lot_metrics = evaluate_on_test(
        lot_best, lot_yaml, out_dir, run_name="split_lot_test",
        imgsz=train_cfg["imgsz"], batch=train_cfg["batch"], class_names=YOLO_CLASS_NAMES,
    )

    print("\n[5/5] Random vs Date vs Lot 비교")
    random_metrics = EvalMetrics(
        mAP50=args.random_map50,
        mAP50_95=args.random_map50_95,
        precision=args.random_precision,
        recall=args.random_recall,
        per_class_mAP50={},
    )
    results = {"random": random_metrics, "date": date_metrics, "lot": lot_metrics}
    df = compare_splits(results)
    df.to_csv(out_dir / "split_comparison.csv", index=False)
    plot_split_comparison(df, save_path=out_dir / "split_comparison.png")

    print(df.to_string(index=False))
    print(f"\n저장 위치: {out_dir}/split_comparison.csv, split_comparison.png")


if __name__ == "__main__":
    main()
