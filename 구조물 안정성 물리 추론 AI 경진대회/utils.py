"""
utils.py — 공통 유틸리티
"""

import random
import numpy as np
import torch


def seed_everything(seed: int):
    """재현성을 위한 전역 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
