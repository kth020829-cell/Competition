"""
inference.py — 추론 및 TTA(Test Time Augmentation) 앙상블
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CKPT_DIR, device, CFG
from dataset import MultiViewDataset, get_test_transform, get_tta_transform
from model import MultiViewModelV2


@torch.no_grad()
def predict_single(model, df, data_dir, transform, cfg) -> np.ndarray:
    """
    단일 모델 + 단일 transform으로 추론.

    Returns:
        (N, 2) 확률 배열
    """
    ds = MultiViewDataset(df, data_dir, transform=transform, is_test=True)
    loader = DataLoader(ds, batch_size=cfg['batch_size'],
                        shuffle=False, num_workers=cfg['num_workers'],
                        pin_memory=True)
    probs_list = []
    for views in loader:
        views = [v.to(device) for v in views]
        with torch.amp.autocast('cuda'):
            logits = model(views)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(p)
    return np.vstack(probs_list)


def run_inference(seeds: list, cfg: dict, test_df,
                  data_dir, ckpt_prefix: str = 'best_model_seed') -> np.ndarray:
    """
    Multi-Seed × TTA 앙상블 추론.

    전략:
        각 seed 모델에 대해 원본 추론 1회 + TTA tta_num회 실행 후 평균.
        최종 예측은 모든 seed 모델 예측의 평균.

    Args:
        seeds       : 앙상블할 seed 리스트
        cfg         : CFG 딕셔너리
        test_df     : test DataFrame
        data_dir    : test 이미지 경로
        ckpt_prefix : 체크포인트 파일명 prefix

    Returns:
        (N, 2) 정규화된 최종 확률 배열
    """
    all_preds = []

    for seed in seeds:
        ckpt = CKPT_DIR / f'{ckpt_prefix}{seed}.pth'
        if not ckpt.exists():
            print(f'⚠️  {ckpt} 없음 — skip')
            continue

        model = MultiViewModelV2(
            cfg['model_name'], freeze_encoder=False, dropout=cfg['dropout']
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        print(f'[seed={seed}] 추론 시작...')

        # 원본 + TTA
        preds = [predict_single(model, test_df, data_dir,
                                get_test_transform(cfg['img_size']), cfg)]
        for t in range(cfg['tta_num']):
            preds.append(predict_single(model, test_df, data_dir,
                                        get_tta_transform(cfg['img_size']), cfg))
            print(f'  TTA {t+1}/{cfg["tta_num"]} 완료')

        all_preds.append(np.mean(preds, axis=0))
        del model
        torch.cuda.empty_cache()

    final = np.mean(all_preds, axis=0)
    return final / final.sum(axis=1, keepdims=True)
