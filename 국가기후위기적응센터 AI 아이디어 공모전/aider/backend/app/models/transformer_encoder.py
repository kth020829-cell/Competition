"""
경량 Transformer 인코더 — Bt (생체 반응 점수) 시계열 인코딩.
PatchTST/iTransformer 계열 아이디어: patch embedding + self-attention 2층.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from app.config import get_settings

FEATURE_DIM = 4      # hr, hrv, skin_temp, spo2
PATCH_LEN = 6        # 6분 구간 하나의 패치
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
MAX_PATCHES = 20     # 최대 시퀀스 길이 (패치 수)


class PatchEmbedding(nn.Module):
    def __init__(self, feature_dim: int, patch_len: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(feature_dim * patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_patches, patch_len, feature_dim)
        B, N, P, F = x.shape
        x = x.view(B, N, P * F)
        return self.proj(x)  # (B, N, d_model)


class BioTransformerEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        patch_len: int = PATCH_LEN,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.patch_embed = PatchEmbedding(feature_dim, patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feature_dim) where T = n_patches * patch_len
        B, T, F = x.shape
        n_patches = T // self.patch_len
        x = x[:, : n_patches * self.patch_len, :]
        x = x.view(B, n_patches, self.patch_len, F)
        x = self.patch_embed(x)          # (B, n_patches, d_model)
        x = self.encoder(x)              # (B, n_patches, d_model)
        x = x.permute(0, 2, 1)          # (B, d_model, n_patches)
        x = self.pool(x).squeeze(-1)    # (B, d_model)
        return self.head(x).squeeze(-1)  # (B,)


_transformer_model: BioTransformerEncoder | None = None


def _get_model_path() -> Path:
    return Path(get_settings().model_artifact_dir) / "transformer_bt.pt"


def load_transformer() -> BioTransformerEncoder | None:
    global _transformer_model
    path = _get_model_path()
    if path.exists():
        model = BioTransformerEncoder()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _transformer_model = model
    return _transformer_model


def get_transformer() -> BioTransformerEncoder | None:
    global _transformer_model
    if _transformer_model is None:
        load_transformer()
    return _transformer_model


def encode_bio_sequence(df_wearable) -> float:
    """
    DataFrame(columns: hr_bpm, hrv_ms, skin_temp_c, spo2_pct) → Bt 점수 [0,1].
    모델 없으면 통계 기반 fallback 반환.
    """
    model = get_transformer()

    arr = df_wearable[["hr_bpm", "hrv_ms", "skin_temp_c", "spo2_pct"]].values.astype(np.float32)

    # 정규화 (학습 시 통계 기반; 여기서는 고정 스케일)
    means = np.array([72.0, 55.0, 36.6, 98.0])
    stds = np.array([15.0, 20.0, 0.8, 2.0])
    arr = (arr - means) / stds

    if model is None or len(arr) < PATCH_LEN:
        # 통계 기반 fallback
        from app.services.feature_engineering import BioFeatures, compute_bt
        last = df_wearable.iloc[-1]
        bio = BioFeatures(
            hr_bpm=float(last["hr_bpm"]),
            hrv_ms=float(last["hrv_ms"]),
            skin_temp_c=float(last["skin_temp_c"]),
            spo2_pct=float(last["spo2_pct"]),
        )
        return compute_bt(bio)

    # 패딩/트리밍
    max_len = MAX_PATCHES * PATCH_LEN
    if len(arr) > max_len:
        arr = arr[-max_len:]
    padded = np.zeros((max_len, FEATURE_DIM), dtype=np.float32)
    padded[-len(arr):] = arr

    x = torch.tensor(padded).unsqueeze(0)  # (1, max_len, 4)
    with torch.no_grad():
        bt = model(x).item()
    return float(np.clip(bt, 0.0, 1.0))


def save_transformer(model: BioTransformerEncoder):
    path = _get_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
