"""
Transformer 인코더 (Bt) 학습 스크립트
스트레스 레이블(0=없음, 1=있음)을 지도학습으로 예측 → Bt 점수로 활용
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from app.services.synthetic_generator import WearableSimulator
from app.models.transformer_encoder import (
    BioTransformerEncoder, FEATURE_DIM, PATCH_LEN, MAX_PATCHES,
    save_transformer,
)

SEED = 42
N_PER_CLASS = 1000
EPOCHS = 30
BATCH_SIZE = 32
LR = 5e-4
SEQ_LEN = MAX_PATCHES * PATCH_LEN


def build_dataset():
    sim = WearableSimulator()
    samples, labels = [], []

    for stress_type, label in [("none", 0), ("heat", 1), ("cold", 1)]:
        n = N_PER_CLASS
        for i in range(n):
            df = sim.generate(stress_type=stress_type, duration_min=SEQ_LEN, seed=SEED + i * 3 + label)
            arr = df[["hr_bpm", "hrv_ms", "skin_temp_c", "spo2_pct"]].values.astype(np.float32)

            means = np.array([72.0, 55.0, 36.6, 98.0], dtype=np.float32)
            stds = np.array([15.0, 20.0, 0.8, 2.0], dtype=np.float32)
            arr = (arr - means) / stds

            if len(arr) >= SEQ_LEN:
                arr = arr[:SEQ_LEN]
            else:
                padded = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
                padded[:len(arr)] = arr
                arr = padded

            samples.append(arr)
            labels.append(float(label))

    X = np.stack(samples)
    y = np.array(labels, dtype=np.float32)
    return X, y


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("데이터 생성 중...")
    X, y = build_dataset()

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    tr_set = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_set = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = BioTransformerEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_state = None

    print(f"학습 시작 ({EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for X_b, y_b in tr_loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(X_b)
        tr_loss /= len(X_tr)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                pred = model(X_b)
                val_loss += criterion(pred, y_b).item() * len(X_b)
                correct += ((pred > 0.5).float() == y_b).sum().item()
        val_loss /= len(X_val)
        acc = correct / len(X_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {tr_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.3f}")

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    save_transformer(model)
    print(f"Transformer 저장 완료 (Best Val Loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    train()
