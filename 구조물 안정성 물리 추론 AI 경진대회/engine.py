"""
engine.py — 학습 및 검증 함수
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def compute_logloss(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    대회 공식 2-class LogLoss.

    Args:
        probs  : (N, 2) 확률 배열
        labels : (N,) 정수 라벨 배열
    Returns:
        LogLoss 스칼라 값
    """
    eps = 1e-15
    p = np.clip(probs, eps, 1 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    labels_oh = np.eye(2)[labels]
    return -np.mean(np.sum(labels_oh * np.log(p), axis=1))


def train_one_epoch(model, loader, criterion, optimizer,
                    scheduler, device, scaler) -> float:
    """
    1 epoch 학습.
    Mixed Precision (AMP) + Gradient Clipping 적용.

    Returns:
        평균 train loss
    """
    model.train()
    total_loss = 0.0

    for views, labels in tqdm(loader, desc='Train', leave=False):
        views  = [v.to(device) for v in views]
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(views)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device) -> tuple:
    """
    검증 루프.

    Returns:
        (logloss, accuracy) 튜플
    """
    model.eval()
    all_probs, all_labels = [], []

    for views, labels in tqdm(loader, desc='Val  ', leave=False):
        views = [v.to(device) for v in views]
        with torch.amp.autocast('cuda'):
            logits = model(views)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())

    probs_arr  = np.vstack(all_probs)
    labels_arr = np.array(all_labels)
    logloss    = compute_logloss(probs_arr, labels_arr)
    acc        = (probs_arr.argmax(1) == labels_arr).mean()
    return logloss, acc
