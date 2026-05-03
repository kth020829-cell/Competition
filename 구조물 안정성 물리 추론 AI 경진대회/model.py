"""
model.py — ConvNeXt-Large 기반 Multi-View 구조물 안정성 예측 모델

Architecture:
    [front.png] → ConvNeXt-Large → f1 (1536d)
                                              ↘
                              concat(f1, f2, |f1-f2|, f1*f2) → 6144d
                                              ↗
    [top.png]   → ConvNeXt-Large → f2 (1536d)
                                              ↓
                                  MLP Head (6144 → 1024 → 256 → 2)
                                              ↓
                                  [stable_prob, unstable_prob]
"""

import torch
import torch.nn as nn
import timm


class MultiViewModelV2(nn.Module):
    """
    ConvNeXt-Large 백본 + 4-way Fusion Head.

    Fusion 전략:
        concat(f1, f2, |f1-f2|, f1*f2) → feat_dim * 4
        - f1, f2      : 각 뷰 자체 특징
        - |f1-f2|     : 두 뷰의 비대칭성 (기울기, 하중 불균형 검출)
        - f1 * f2     : 두 뷰의 공통 패턴 강조 (유사도)

    단순 concat 대비 두 뷰의 관계를 명시적으로 학습하여
    경계(Boundary) 샘플 분류 성능을 개선.

    Args:
        model_name     : timm 모델 이름
        num_classes    : 출력 클래스 수 (기본 2: stable / unstable)
        dropout        : Dropout 비율
        freeze_encoder : True면 초기 Head만 학습 (Phase 1)
    """
    def __init__(self, model_name: str, num_classes: int = 2,
                 dropout: float = 0.3, freeze_encoder: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True,
            num_classes=0, global_pool='avg'
        )
        feat_dim  = self.backbone.num_features  # ConvNeXt-Large: 1536
        fused_dim = feat_dim * 4                # 6144

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, views: list) -> torch.Tensor:
        f1 = self.encode(views[0])
        f2 = self.encode(views[1])
        fused = torch.cat([
            f1,
            f2,
            torch.abs(f1 - f2),
            f1 * f2,
        ], dim=1)
        return self.head(fused)

    def unfreeze(self):
        """Phase 2 전환: Encoder 전체 파인튜닝 활성화"""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print('Encoder 전체 파인튜닝 시작')
