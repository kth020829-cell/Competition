"""
pseudo_label.py — Pseudo Label 생성 및 재학습

전략:
    1차 추론에서 모델이 확신하는 test 샘플(threshold=0.95)을
    학습 데이터에 추가하여 재학습 → 데이터 부족 문제 완화.

    과적합 방지:
        - 높은 threshold(0.95)로 불확실한 샘플 제외
        - 낮은 lr(5e-5)로 기존 지식을 유지하며 미세 조정
        - PL 모델은 기존 체크포인트와 별도 저장
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CKPT_DIR, LABEL2IDX, device
from dataset import MixedDataset, get_train_transform, get_test_transform
from engine import train_one_epoch, validate
from model import MultiViewModelV2
from utils import seed_everything


def generate_pseudo_labels(final_probs: np.ndarray,
                            test_df: pd.DataFrame,
                            threshold: float) -> pd.DataFrame:
    """
    모델이 확신하는 test 샘플에 pseudo label 부여.

    Args:
        final_probs : (N, 2) 추론 확률 배열
        test_df     : test DataFrame (id 컬럼 포함)
        threshold   : 확신 기준 (기본 0.95)

    Returns:
        pseudo label이 부여된 DataFrame
    """
    unstable_probs  = final_probs[:, 1]
    pl_unstable_idx = np.where(unstable_probs > threshold)[0]
    pl_stable_idx   = np.where(unstable_probs < (1 - threshold))[0]

    print(f'Threshold      : {threshold}')
    print(f'Pseudo unstable: {len(pl_unstable_idx)}개')
    print(f'Pseudo stable  : {len(pl_stable_idx)}개')
    print(f'총 Pseudo Label: {len(pl_unstable_idx) + len(pl_stable_idx)}개')

    pl_u          = test_df.iloc[pl_unstable_idx].copy()
    pl_u['label'] = 'unstable'
    pl_u['split'] = 'test'

    pl_s          = test_df.iloc[pl_stable_idx].copy()
    pl_s['label'] = 'stable'
    pl_s['split'] = 'test'

    pseudo_df = pd.concat([pl_u, pl_s], ignore_index=True)
    print(f'\nPseudo Label 분포:\n{pseudo_df["label"].value_counts()}')
    return pseudo_df


def run_pseudo_training(seed: int, cfg: dict,
                        pseudo_df: pd.DataFrame,
                        train_df: pd.DataFrame,
                        dev_df: pd.DataFrame) -> float:
    """
    기존 체크포인트에서 이어서 Pseudo Label 포함 재학습.

    Args:
        seed      : 랜덤 시드
        cfg       : CFG 딕셔너리
        pseudo_df : generate_pseudo_labels()의 반환값
        train_df  : 원본 train DataFrame
        dev_df    : 원본 dev DataFrame

    Returns:
        best val logloss
    """
    seed_everything(seed)
    use_dp = torch.cuda.device_count() > 1
    print(f'\n{"="*60}')
    print(f' Pseudo Label 재학습 | Seed {seed}')
    print(f'{"="*60}')

    _train = train_df.copy(); _train['split'] = 'train'
    _dev   = dev_df.copy();   _dev['split']   = 'dev'
    all_df = pd.concat([_train, _dev, pseudo_df], ignore_index=True)
    _val   = dev_df.copy();   _val['split']   = 'dev'

    print(f'학습 데이터: {len(all_df)}개 '
          f'(기존 {len(_train)+len(_dev)}개 + Pseudo {len(pseudo_df)}개)')

    train_ds = MixedDataset(all_df, transform=get_train_transform(cfg['img_size']))
    val_ds   = MixedDataset(_val,   transform=get_test_transform(cfg['img_size']))

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'],
                              pin_memory=True)

    # 기존 best 모델에서 이어서 학습
    ckpt  = CKPT_DIR / f'best_model_seed{seed}.pth'
    model = MultiViewModelV2(
        cfg['model_name'], freeze_encoder=False, dropout=cfg['dropout']
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f'체크포인트 로드: {ckpt}')

    if use_dp:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg['pl_lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['pl_epochs'], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda')

    best_logloss, best_epoch = float('inf'), 0

    for epoch in range(1, cfg['pl_epochs'] + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        val_logloss, val_acc = validate(model, val_loader, device)

        flag = ''
        if val_logloss < best_logloss:
            best_logloss, best_epoch = val_logloss, epoch
            state = model.module.state_dict() if use_dp else model.state_dict()
            # 기존 체크포인트 덮어쓰기 방지 — pl_ prefix로 별도 저장
            torch.save(state, CKPT_DIR / f'best_model_pl_seed{seed}.pth')
            flag = ' ★'

        print(f'  [PL {epoch:02d}/{cfg["pl_epochs"]}] '
              f'tr={train_loss:.4f} '
              f'vl={val_logloss:.4f} '
              f'acc={val_acc:.4f}{flag}')

    print(f'  → Seed {seed} PL 완료: best ep={best_epoch}, logloss={best_logloss:.4f}')
    return best_logloss
