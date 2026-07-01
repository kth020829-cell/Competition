"""
오토인코더 이상탐지 학습 스크립트
정상(none stress) 시계열만 학습 → 재구성 오차 임계값 설정
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.services.synthetic_generator import WearableSimulator
from app.models.autoencoder import TimeSeriesAutoEncoder, SEQ_LEN, FEATURE_DIM, save_autoencoder

SEED = 42
N_NORMAL = 2000
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
THRESHOLD_PERCENTILE = 95   # 정상 데이터 재구성 오차의 95퍼센타일을 임계값으로


def build_dataset(n: int, stress_type: str, seed_offset: int = 0):
    sim = WearableSimulator()
    samples = []
    for i in range(n):
        df = sim.generate(stress_type=stress_type, duration_min=60, seed=seed_offset + i)
        arr = df[["hr_bpm", "hrv_ms", "skin_temp_c", "spo2_pct"]].values.astype(np.float32)

        # 정규화
        means = np.array([72.0, 55.0, 36.6, 98.0], dtype=np.float32)
        stds = np.array([15.0, 20.0, 0.8, 2.0], dtype=np.float32)
        arr = (arr - means) / stds

        # 길이 고정
        if len(arr) >= SEQ_LEN:
            arr = arr[:SEQ_LEN]
        else:
            padded = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
            padded[:len(arr)] = arr
            arr = padded

        samples.append(arr)

    return np.stack(samples)  # (N, SEQ_LEN, FEATURE_DIM)


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("정상 데이터 생성 중...")
    X_normal = build_dataset(N_NORMAL, "none", seed_offset=0)
    X_train = torch.tensor(X_normal)

    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TimeSeriesAutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"학습 시작 ({EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        avg_loss = total_loss / len(X_normal)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.5f}")

    # 임계값 설정: 정상 데이터 재구성 오차의 95퍼센타일
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(X_train).numpy()
    threshold = float(np.percentile(errors, THRESHOLD_PERCENTILE))
    print(f"임계값 (p{THRESHOLD_PERCENTILE}): {threshold:.4f}")

    # 이상 데이터 검증
    print("이상 데이터 검증...")
    X_heat = torch.tensor(build_dataset(200, "heat", seed_offset=9000))
    X_cold = torch.tensor(build_dataset(200, "cold", seed_offset=9200))
    with torch.no_grad():
        heat_errors = model.reconstruction_error(X_heat).numpy()
        cold_errors = model.reconstruction_error(X_cold).numpy()

    print(f"정상 평균 오차: {errors.mean():.4f}")
    print(f"폭염 평균 오차: {heat_errors.mean():.4f} (탐지율 {(heat_errors > threshold).mean():.1%})")
    print(f"한파 평균 오차: {cold_errors.mean():.4f} (탐지율 {(cold_errors > threshold).mean():.1%})")

    model.eval()
    save_autoencoder(model, threshold)
    print("오토인코더 저장 완료")


if __name__ == "__main__":
    train()
