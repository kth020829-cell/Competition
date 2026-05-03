"""
dataset.py — Dataset 및 Transform 정의

Transform 설계 원칙:
    Train(고정 조명/카메라) → Test(랜덤 조명/카메라) 도메인 갭을 해결하기 위해
    강한 augmentation으로 다양한 환경 변화를 시뮬레이션.
"""

from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import TRAIN_DIR, DEV_DIR, TEST_DIR, LABEL2IDX


# =============================================================================
# Transform
# =============================================================================
def get_train_transform(img_size: int):
    """
    도메인 갭 대응 Augmentation.
    - RandomPerspective : 카메라 시점 변화 시뮬레이션
    - ColorJitter       : 조명/색상 변화 시뮬레이션
    - GaussianBlur      : 포커스 변화 시뮬레이션
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])


def get_test_transform(img_size: int):
    """검증/추론용 — 기본 Resize + Normalize"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_tta_transform(img_size: int):
    """TTA(Test Time Augmentation)용 — 가벼운 랜덤 변환"""
    return transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# =============================================================================
# Dataset
# =============================================================================
class MultiViewDataset(Dataset):
    """
    front.png + top.png 두 시점 이미지 데이터셋.

    Args:
        df       : id, label 컬럼을 포함한 DataFrame
        data_dir : 이미지 루트 디렉토리
        transform: torchvision transform
        is_test  : True면 label 없이 이미지만 반환 (추론용)
    """
    def __init__(self, df, data_dir: Path,
                 transform=None, is_test: bool = False):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.is_test   = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        folder = self.data_dir / row['id']
        views  = []
        for name in ['front', 'top']:
            img = Image.open(folder / f'{name}.png').convert('RGB')
            if self.transform:
                img = self.transform(img)
            views.append(img)
        if self.is_test:
            return views
        return views, LABEL2IDX[row['label']]


class MixedDataset(Dataset):
    """
    train + dev + pseudo(test) 혼합 데이터셋.
    split 컬럼('train' | 'dev' | 'test')으로 이미지 경로를 분기.

    Args:
        df       : id, label, split 컬럼을 포함한 DataFrame
        transform: torchvision transform
    """
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        split = row.get('split', 'train')

        if split == 'dev':
            folder = DEV_DIR / row['id']
        elif split == 'test':
            folder = TEST_DIR / row['id']
        else:
            folder = TRAIN_DIR / row['id']

        views = []
        for name in ['front', 'top']:
            img = Image.open(folder / f'{name}.png').convert('RGB')
            if self.transform:
                img = self.transform(img)
            views.append(img)

        return views, LABEL2IDX[row['label']]
