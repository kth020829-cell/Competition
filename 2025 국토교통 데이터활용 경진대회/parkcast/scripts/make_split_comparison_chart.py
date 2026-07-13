"""README의 Random/Date/Lot split 결과(이미 학습·평가 완료된 숫자)로 비교 막대그래프만 재생성.

재학습 없이 포트폴리오용 그래프가 필요할 때 씀. 실제 재현 파이프라인은
scripts/cross_lot_eval.py (매번 새로 학습) 참조.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from parkcast.domain import compare_splits, plot_split_comparison
from parkcast.evaluate import EvalMetrics

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

results = {
    "Random": EvalMetrics(mAP50=0.9944, mAP50_95=0.9886, precision=0.9977, recall=0.9975, per_class_mAP50={}),
    "Date": EvalMetrics(mAP50=0.995, mAP50_95=0.805, precision=0.995, recall=0.995, per_class_mAP50={}),
    "Lot": EvalMetrics(mAP50=0.995, mAP50_95=0.989, precision=0.999, recall=0.998, per_class_mAP50={}),
}

df = compare_splits(results)
out_dir = Path(__file__).resolve().parent.parent / "results" / "cross_lot"
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(out_dir / "split_comparison.csv", index=False)
plot_split_comparison(df, save_path=out_dir / "split_comparison.png")

print(df.to_string(index=False))
print(f"\n저장 위치: {out_dir / 'split_comparison.png'}")
