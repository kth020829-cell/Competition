"""
train.py — 전체 학습 파이프라인 실행 진입점

실행 방법:
    python train.py

실행 전 config.py에서 경로 설정:
    ROOT     = Path('./data')        # 데이터 경로
    CKPT_DIR = Path('./checkpoints') # 체크포인트 저장 경로
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import ROOT, TRAIN_DIR, DEV_DIR, TEST_DIR, CKPT_DIR, CFG, device
from dataset import MixedDataset, get_train_transform, get_test_transform
from engine import train_one_epoch, validate
from inference import run_inference
from model import MultiViewModelV2
from pseudo_label import generate_pseudo_labels, run_pseudo_training
from utils import seed_everything


def run_training(seed: int, cfg: dict,
                 train_df: pd.DataFrame,
                 dev_df: pd.DataFrame) -> tuple:
    """
    단일 seed 학습.

    2단계 학습 전략:
        Phase 1 (epoch 1 ~ unfreeze_epoch)    : Head만 학습 → 빠른 수렴
        Phase 2 (epoch unfreeze_epoch+1 ~ end): Encoder 전체 파인튜닝 (lr × 0.1)

    Args:
        seed     : 랜덤 시드
        cfg      : CFG 딕셔너리
        train_df : train DataFrame
        dev_df   : dev DataFrame (validation)

    Returns:
        (best_logloss, history DataFrame) 튜플
    """
    seed_everything(seed)
    use_dp = torch.cuda.device_count() > 1
    print(f'\n{"="*60}')
    print(f' Seed {seed} 학습 시작 | GPU {torch.cuda.device_count()}개')
    print(f'{"="*60}')

    # train + dev 합쳐서 학습, dev만 val로 사용
    _train = train_df.copy(); _train['split'] = 'train'
    _dev   = dev_df.copy();   _dev['split']   = 'dev'
    all_df = pd.concat([_train, _dev], ignore_index=True)
    _val   = dev_df.copy();   _val['split']   = 'dev'

    train_ds = MixedDataset(all_df, transform=get_train_transform(cfg['img_size']))
    val_ds   = MixedDataset(_val,   transform=get_test_transform(cfg['img_size']))

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'],
                              pin_memory=True)

    model = MultiViewModelV2(
        cfg['model_name'], freeze_encoder=True, dropout=cfg['dropout']
    ).to(device)
    if use_dp:
        model = nn.DataParallel(model)
        print(f'DataParallel: GPU {torch.cuda.device_count()}개 사용')

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'],
        total_steps=cfg['epochs'] * len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler('cuda')

    best_logloss, best_epoch, history = float('inf'), 0, []

    for epoch in range(1, cfg['epochs'] + 1):

        # Phase 2 전환
        if epoch == cfg['unfreeze_epoch'] + 1:
            base = model.module if use_dp else model
            base.unfreeze()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg['lr'] * 0.1, weight_decay=cfg['weight_decay']
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg['epochs'] - epoch, eta_min=1e-6
            )

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        val_logloss, val_acc = validate(model, val_loader, device)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_logloss': val_logloss, 'val_acc': val_acc})

        flag = ''
        if val_logloss < best_logloss:
            best_logloss, best_epoch = val_logloss, epoch
            state = model.module.state_dict() if use_dp else model.state_dict()
            torch.save(state, CKPT_DIR / f'best_model_seed{seed}.pth')
            flag = ' ★ saved'

        print(f'[{epoch:02d}/{cfg["epochs"]}] '
              f'train={train_loss:.4f}  '
              f'val_ll={val_logloss:.4f}  '
              f'val_acc={val_acc:.4f}{flag}')

    print(f'\nSeed {seed} 완료 → Best ep={best_epoch}, LogLoss={best_logloss:.4f}')
    return best_logloss, pd.DataFrame(history)


def main():
    # 환경 출력
    print(f'Device: {device}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    train_df = pd.read_csv(ROOT / 'train.csv')
    dev_df   = pd.read_csv(ROOT / 'dev.csv')
    test_df  = pd.read_csv(ROOT / 'sample_submission.csv')

    print(f'\nTrain : {len(train_df)}개')
    print(f'Dev   : {len(dev_df)}개')
    print(f'Test  : {len(test_df)}개')
    print(f'\nTrain label 분포:\n{train_df["label"].value_counts()}')

    # ── Phase 1: 기본 학습 ───────────────────────────────────────────────────
    all_results, all_histories = {}, {}
    for seed in CFG['seeds']:
        ll, hist = run_training(seed, CFG, train_df, dev_df)
        all_results[seed]   = ll
        all_histories[seed] = hist

    print('\n' + '='*60)
    print('📊 기본 학습 결과')
    for s, ll in all_results.items():
        print(f'  seed={s}: {ll:.4f}')
    print(f'  평균: {np.mean(list(all_results.values())):.4f}')

    # ── Phase 2: 1차 추론 ────────────────────────────────────────────────────
    final_probs = run_inference(CFG['seeds'], CFG, test_df, TEST_DIR)
    print(f'\n1차 추론 완료 | unstable 평균: {final_probs[:,1].mean():.4f}')

    # ── Phase 3: Pseudo Label 재학습 ─────────────────────────────────────────
    pseudo_df = generate_pseudo_labels(final_probs, test_df, CFG['pl_threshold'])

    pl_results = {}
    for seed in CFG['seeds']:
        ll = run_pseudo_training(seed, CFG, pseudo_df, train_df, dev_df)
        pl_results[seed] = ll

    print('\n' + '='*60)
    print('📊 Pseudo Label 재학습 결과')
    for s, ll in pl_results.items():
        print(f'  seed={s}: {ll:.4f}')
    print(f'  평균: {np.mean(list(pl_results.values())):.4f}')

    # ── Phase 4: 최종 추론 & 제출 파일 생성 ─────────────────────────────────
    final_probs_pl = run_inference(
        CFG['seeds'], CFG, test_df, TEST_DIR,
        ckpt_prefix='best_model_pl_seed'
    )

    submission = pd.DataFrame({
        'id'           : test_df['id'],
        'unstable_prob': final_probs_pl[:, 1],
        'stable_prob'  : final_probs_pl[:, 0],
    })

    prob_sum = submission['unstable_prob'] + submission['stable_prob']
    assert np.allclose(prob_sum, 1.0, atol=1e-4), '확률 합 오류!'

    submission.to_csv('submission.csv', index=False, encoding='utf-8-sig')
    print('\n✅ submission.csv 저장 완료')
    print(submission.head())
    print(f'\nunstable_prob 분포:')
    print(submission['unstable_prob'].describe())


if __name__ == '__main__':
    main()
