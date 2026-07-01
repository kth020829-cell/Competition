"""
경량 오토인코더 기반 이상탐지 모듈
- 정상 생체신호 시계열에서 학습 → 재구성 오차로 급변 패턴 포착
- LightGBM CHAI 스코어를 대체하는 게 아니라 추가 신호로 병행
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from app.config import get_settings

FEATURE_DIM = 4   # hr, hrv, skin_temp, spo2
SEQ_LEN = 60      # 1시간 (1분 간격)
LATENT_DIM = 8


class TimeSeriesAutoEncoder(nn.Module):
    def __init__(self, seq_len: int = SEQ_LEN, feature_dim: int = FEATURE_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        input_dim = seq_len * feature_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, feature_dim)
        B = x.shape[0]
        flat = x.reshape(B, -1)
        z = self.encoder(flat)
        recon = self.decoder(z)
        return recon.reshape(B, self.seq_len, self.feature_dim)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))  # (B,)


_ae_model: TimeSeriesAutoEncoder | None = None
_ae_threshold: float = 0.15  # 학습 후 업데이트


def _get_model_path() -> Path:
    return Path(get_settings().model_artifact_dir) / "autoencoder.pt"


def _get_threshold_path() -> Path:
    return Path(get_settings().model_artifact_dir) / "ae_threshold.npy"


def load_autoencoder() -> TimeSeriesAutoEncoder | None:
    global _ae_model, _ae_threshold
    model_path = _get_model_path()
    threshold_path = _get_threshold_path()

    if model_path.exists():
        model = TimeSeriesAutoEncoder()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        _ae_model = model

    if threshold_path.exists():
        _ae_threshold = float(np.load(threshold_path))

    return _ae_model


def get_autoencoder() -> TimeSeriesAutoEncoder | None:
    global _ae_model
    if _ae_model is None:
        load_autoencoder()
    return _ae_model


def detect_anomaly(df_wearable) -> dict:
    """
    생체신호 시계열 DataFrame → 이상 여부 + 재구성 오차 반환.
    """
    arr = df_wearable[["hr_bpm", "hrv_ms", "skin_temp_c", "spo2_pct"]].values.astype(np.float32)

    # 정규화
    means = np.array([72.0, 55.0, 36.6, 98.0])
    stds = np.array([15.0, 20.0, 0.8, 2.0])
    arr = (arr - means) / stds

    # SEQ_LEN에 맞게 패딩/트리밍
    if len(arr) >= SEQ_LEN:
        arr = arr[-SEQ_LEN:]
    else:
        padded = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        padded[-len(arr):] = arr
        arr = padded

    model = get_autoencoder()
    if model is None:
        return {
            "is_anomaly": False,
            "recon_error": None,
            "threshold": _ae_threshold,
            "model_loaded": False,
        }

    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, SEQ_LEN, 4)
    with torch.no_grad():
        error = model.reconstruction_error(x).item()

    return {
        "is_anomaly": error > _ae_threshold,
        "recon_error": round(error, 4),
        "threshold": round(_ae_threshold, 4),
        "model_loaded": True,
    }


def save_autoencoder(model: TimeSeriesAutoEncoder, threshold: float):
    model_path = _get_model_path()
    threshold_path = _get_threshold_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)
    np.save(threshold_path, np.array(threshold))

    global _ae_model, _ae_threshold
    _ae_model = model
    _ae_threshold = threshold
