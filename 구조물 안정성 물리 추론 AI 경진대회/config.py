"""
config.py — 경로 및 하이퍼파라미터 설정
"""

from pathlib import Path
import torch

# =============================================================================
# 경로 설정
# =============================================================================
ROOT      = Path('./data')        # ★ 실제 데이터 경로로 수정
TRAIN_DIR = ROOT / 'train'
DEV_DIR   = ROOT / 'dev'
TEST_DIR  = ROOT / 'test'
CKPT_DIR  = Path('./checkpoints')
CKPT_DIR.mkdir(exist_ok=True)

# =============================================================================
# 하이퍼파라미터
# =============================================================================
CFG = {
    # 모델
    'model_name'      : 'convnext_large.fb_in22k_ft_in1k',
    'img_size'        : 224,
    'dropout'         : 0.3,

    # 학습
    'epochs'          : 15,
    'batch_size'      : 16,
    'lr'              : 2e-4,
    'weight_decay'    : 1e-4,
    'label_smoothing' : 0.1,
    'unfreeze_epoch'  : 5,
    'num_workers'     : 0,

    # 앙상블
    'seeds'           : [42, 2024, 777],
    'tta_num'         : 5,

    # Pseudo Label
    'pl_threshold'    : 0.95,
    'pl_epochs'       : 8,
    'pl_lr'           : 5e-5,
}

LABEL2IDX = {'stable': 0, 'unstable': 1}

# =============================================================================
# 디바이스
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
