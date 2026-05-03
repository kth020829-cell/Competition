# 구조물 안정성 물리 추론 AI 경진대회

데이콘 월간 대회 — 두 시점 이미지를 기반으로 구조물의 붕괴 확률을 예측하는 AI 모델 개발

## 대회 개요

구조물의 `front.png`, `top.png` 두 시점 이미지를 입력으로 받아, 시뮬레이션 시작 10초 이내에 구조물이 불안정(unstable) 상태로 전환될 확률을 예측합니다.
> 🏆 최종 성적: Private Score **0.08636** | 상위 **23%**
- **평가지표**: LogLoss (낮을수록 좋음)
- **핵심 난이도**: Train(고정 조명/카메라) → Test(랜덤 조명/카메라) 도메인 갭

| 구분 | 데이터 수 | 환경 |
|------|---------|------|
| Train | 1,000개 | 고정 (조명·카메라 고정) |
| Dev | 100개 | 랜덤 (실제 평가 환경과 동일) |
| Test | 1,000개 | 랜덤 |

## 성능

| 버전 | 모델 | 전략 | Public LogLoss | Private LogLoss |
|------|------|------|---------------|----------------|
| Baseline | ResNet-18 | BCE Loss | 1.037 | - |
| v1 | EfficientNet-B2 | TTA | 0.120 | - |
| v2 | ConvNeXt-Base | 4-way Fusion + 3-Seed | 0.047 | - |
| **최종** | **ConvNeXt-Large** | **+ Pseudo Label** | **-** | **0.08636 (상위 23%)** |

## 모델 아키텍처

```
[front.png] ──→ ConvNeXt-Large ──→ f1 (1536d) ──┐
                                                  │  concat(f1, f2, |f1-f2|, f1*f2)
[top.png]   ──→ ConvNeXt-Large ──→ f2 (1536d) ──┘         (6144d)
                                                                │
                                                   MLP Head (6144→1024→256→2)
                                                                │
                                                  [stable_prob, unstable_prob]
```

### 4-way Fusion

단순 `concat(f1, f2)` 대신 두 뷰의 관계를 명시적으로 학습합니다.

| 연산 | 의미 |
|------|------|
| `f1, f2` | 각 뷰 자체 특징 |
| `｜f1 - f2｜` | 두 뷰의 비대칭성 (기울기, 하중 불균형 검출) |
| `f1 * f2` | 두 뷰의 공통 패턴 (유사도) |

## 핵심 전략

### 1. 도메인 갭 대응 Augmentation

고정 환경(Train)과 랜덤 환경(Test)의 도메인 갭을 해결하기 위해 다양한 환경 변화를 시뮬레이션합니다.

```python
transforms.RandomPerspective(distortion_scale=0.3, p=0.4)  # 카메라 시점 변화
transforms.ColorJitter(brightness=0.4, contrast=0.4, ...)   # 조명 변화
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))    # 포커스 변화
```

### 2. 2단계 학습 전략

| 단계 | Epoch | 학습 대상 | 학습률 |
|------|-------|---------|-------|
| Phase 1 | 1 ~ 5 | Head만 | 2e-4 |
| Phase 2 | 6 ~ 15 | Encoder 전체 | 2e-5 |

초기에 Head만 학습하여 빠르게 수렴시킨 뒤, Encoder를 낮은 학습률로 파인튜닝합니다.

### 3. Multi-Seed × TTA 앙상블

- **Multi-Seed**: seed 42, 2024, 777로 학습한 3개 모델의 예측 평균
- **TTA**: 추론 시 원본 + 5회 랜덤 변환 예측 평균
- 최종 예측: **3 seeds × 6 (원본+TTA) = 18개 예측 앙상블**

### 4. Pseudo Label

1차 추론에서 모델이 확신하는 샘플(threshold=0.95)을 학습 데이터에 추가하여 재학습합니다.

```
1차 추론 → unstable_prob > 0.95 → 'unstable' 라벨 부여
         → unstable_prob < 0.05 → 'stable' 라벨 부여
         → 기존 1100개 + pseudo N개로 재학습
```

## 파일 구조

```
dacon-structure-stability/
├── train.py          # 메인 실행 파일 (전체 파이프라인)
├── config.py         # 경로 및 하이퍼파라미터 설정
├── dataset.py        # Dataset, Transform 정의
├── model.py          # MultiViewModelV2 (ConvNeXt-Large + 4-way Fusion)
├── engine.py         # 학습/검증 함수
├── inference.py      # 추론, TTA 앙상블
├── pseudo_label.py   # Pseudo Label 생성 및 재학습
└── utils.py          # 공통 유틸리티 (seed 고정 등)
```

## 실행 환경

```
Python  3.10
PyTorch 2.x
timm    latest
GPU     Kaggle T4 x2 (학습 약 11시간)
```

```bash
pip install torch torchvision timm tqdm pandas pillow
```

## 실행 방법

```bash
# 1. config.py에서 데이터 경로 수정
ROOT     = Path('./data')
CKPT_DIR = Path('./checkpoints')

# 2. 실행
python train.py
```

## 참고

- [ConvNeXt 논문](https://arxiv.org/abs/2201.03545) — 백본 선택 근거
- [데이콘 대회 페이지](https://dacon.io) — 대회 설명 및 데이터
