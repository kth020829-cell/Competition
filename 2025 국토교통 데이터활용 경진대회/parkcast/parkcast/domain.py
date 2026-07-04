"""Cross-lot 도메인 갭 평가 — 이미지 임베딩 클러스터링 기반 주차장 자동 발견 + split 비교.

`notebooks/ParkCast_Week2_CrossLot.ipynb`에서 사용하는 재사용 가능한 함수 모음.
PKLot(Roboflow v2)의 random split은 같은 카메라 시점 사진이
train/test 양쪽에 섞여 있어 mAP가 진짜 일반화 성능을 반영하지 못함(README 참조).
파일명에 주차장 정보가 없으므로, ImageNet pretrained ResNet50 임베딩을 K-Means로
클러스터링해 "주차장(lot)"을 라벨 없이 자동 발견하고, 한 클러스터를 통째로 떼어내
"본 적 없는 주차장"으로 평가하는 cross-lot split을 구성한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data import _link_or_copy

if TYPE_CHECKING:
    from .evaluate import EvalMetrics

# Roboflow PKLot 파일명 패턴: '2013-03-22_12_55_08_jpg.rf.해시.jpg'
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})")


def parse_filename_datetime(fname: str) -> Optional[Dict[str, object]]:
    """파일명에서 날짜/시(hour)를 추출. 매치 안 되면 None."""
    m = DATE_PATTERN.search(fname)
    if not m:
        return None
    date, hh, mm, ss = m.groups()
    return {"date": date, "hour": int(hh), "time": f"{hh}:{mm}:{ss}"}


def build_image_metadata(
    yolo_root: str | Path, splits: Tuple[str, ...] = ("train", "valid", "test")
) -> pd.DataFrame:
    """yolo_root(prepare_data.py 산출물) 아래 모든 이미지의 경로/원래 split/날짜를 모음."""
    yolo_root = Path(yolo_root)
    rows = []
    for split in splits:
        img_dir = yolo_root / split / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            parsed = parse_filename_datetime(img_path.name)
            rows.append({
                "fname": img_path.name,
                "path": str(img_path),
                "orig_split": split,
                "date": parsed["date"] if parsed else None,
                "hour": parsed["hour"] if parsed else None,
            })
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def extract_resnet50_embeddings(
    image_paths: List[str], device: Optional[str] = None, batch_size: int = 64
) -> np.ndarray:
    """ImageNet pretrained ResNet50 backbone(avgpool 직전, 2048-dim) → (N, 2048) 임베딩."""
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()

    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class _ImageOnlyDataset(Dataset):
        def __init__(self, paths: List[str]) -> None:
            self.paths = paths

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int):
            img = Image.open(self.paths[idx]).convert("RGB")
            return tf(img)

    loader = DataLoader(_ImageOnlyDataset(image_paths), batch_size=batch_size, num_workers=2)

    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ResNet50 embedding"):
            out = resnet(batch.to(device))
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


@dataclass
class ClusterResult:
    labels: np.ndarray
    best_k: int
    silhouette_scores: Dict[int, float]
    umap_xy: Optional[np.ndarray] = None


def discover_lots(
    embeddings: np.ndarray,
    k_range: Tuple[int, int] = (2, 7),
    seed: int = 0,
    with_umap: bool = True,
) -> ClusterResult:
    """PCA(50) → K-Means(k=2~6, silhouette 최댓값 선택) → (선택) UMAP 2D로 도메인 자동 발견."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(embeddings)
    X_pca = PCA(n_components=50, random_state=seed).fit_transform(X)

    scores: Dict[int, float] = {}
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(*k_range):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_pca)
        s = silhouette_score(X_pca, km.labels_, sample_size=min(2000, len(X_pca)), random_state=seed)
        scores[k] = s
        if s > best_score:
            best_k, best_score, best_labels = k, s, km.labels_

    umap_xy = None
    if with_umap:
        import umap

        umap_xy = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed
        ).fit_transform(X_pca)

    return ClusterResult(labels=best_labels, best_k=best_k, silhouette_scores=scores, umap_xy=umap_xy)


def assign_date_split(meta: pd.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1) -> pd.Series:
    """날짜 기준 시간순 분리: 앞 train_frac 기간 train, 다음 val_frac val, 나머지 test."""
    dates_sorted = sorted(meta["date"].dropna().unique())
    n_train = int(len(dates_sorted) * train_frac)
    n_val = int(len(dates_sorted) * val_frac)
    train_dates = set(dates_sorted[:n_train])
    val_dates = set(dates_sorted[n_train:n_train + n_val])

    def _assign(d):
        if d in train_dates:
            return "train"
        if d in val_dates:
            return "val"
        return "test"

    return meta["date"].map(_assign)


def assign_lot_split(
    meta: pd.DataFrame,
    cluster_col: str = "lot_cluster",
    test_cluster: Optional[int] = None,
    val_frac: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    """클러스터 하나를 통째로 test로 떼어내 "본 적 없는 주차장"으로 평가하는 split 구성.

    test_cluster를 지정 안 하면 표본이 가장 적은 클러스터를 자동 선택.
    """
    if test_cluster is None:
        test_cluster = meta[cluster_col].value_counts().idxmin()

    split = pd.Series("train", index=meta.index)
    is_test = meta[cluster_col] == test_cluster
    split[is_test] = "test"

    non_test_idx = meta.index[~is_test]
    rng = np.random.RandomState(seed)
    n_val = int(len(non_test_idx) * val_frac)
    val_idx = rng.choice(non_test_idx, size=n_val, replace=False)
    split.loc[val_idx] = "val"
    return split


def build_yolo_split(
    meta: pd.DataFrame,
    split_col: str,
    yolo_root: str | Path,
    out_root: str | Path,
    class_names: List[str],
    use_symlink: bool = True,
) -> Path:
    """meta[split_col](train/val/test) 기준으로 새 YOLO 폴더를 구성하고 data.yaml 경로를 반환.

    prepare_data.py가 이미 만들어 둔 yolo_root의 라벨 파일을 재사용함 — 박스/폴리곤
    라벨은 split이 바뀌어도 이미지별로 동일하므로 다시 만들 필요 없이 재배치만 함.
    """
    yolo_root = Path(yolo_root)
    out_root = Path(out_root)

    label_lookup: Dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        lbl_dir = yolo_root / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            label_lookup[lbl.stem] = lbl

    for split_name in ("train", "val", "test"):
        (out_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

    n_missing = 0
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"build split ({out_root.name})"):
        split_name = row[split_col]
        src_img = Path(row["path"])
        stem = src_img.stem

        if stem not in label_lookup:
            n_missing += 1
            continue

        dst_img = out_root / split_name / "images" / src_img.name
        dst_lbl = out_root / split_name / "labels" / f"{stem}.txt"
        if not dst_img.exists():
            _link_or_copy(src_img, dst_img, use_symlink)
        if not dst_lbl.exists():
            _link_or_copy(label_lookup[stem], dst_lbl, use_symlink)

    if n_missing:
        print(f"  경고: 라벨 파일을 못 찾아 건너뛴 이미지 {n_missing}장")

    yaml_content = (
        f"path: {out_root}\n"
        f"train: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: {len(class_names)}\nnames: {class_names}\n"
    )
    data_yaml = out_root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(yaml_content)
    return data_yaml


def compare_splits(results: Dict[str, "EvalMetrics"]) -> pd.DataFrame:
    """{split 이름: EvalMetrics} → 비교용 DataFrame (막대그래프 입력)."""
    rows = [
        {
            "split": name,
            "mAP50": m.mAP50,
            "mAP50_95": m.mAP50_95,
            "precision": m.precision,
            "recall": m.recall,
        }
        for name, m in results.items()
    ]
    return pd.DataFrame(rows)


def plot_split_comparison(df: pd.DataFrame, save_path: Optional[str | Path] = None):
    """Random vs Date vs Lot split 성능 막대그래프. 도메인 갭을 한눈에 보여주는 메인 결과물."""
    import matplotlib.pyplot as plt

    metric_cols = ["mAP50", "mAP50_95", "precision", "recall"]
    x = np.arange(len(df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, col in enumerate(metric_cols):
        ax.bar(x + i * width, df[col], width, label=col)
    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(df["split"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Random vs Date vs Lot split — 도메인 갭 정량 비교")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig
