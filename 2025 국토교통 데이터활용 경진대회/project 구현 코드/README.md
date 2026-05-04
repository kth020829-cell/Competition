# ParkCast Vision

> CCTV/항공 이미지에서 주차장 빈 공간을 실시간으로 검출하는 컴퓨터비전 시스템.
> 한 장의 주차장 이미지를 입력하면 모든 주차칸의 위치 + 점유 여부 + 전체 점유율을 한 번에 출력함.

![demo](docs/demo_preview.png)

---

## 한 줄 요약

YOLOv8 기반으로 PKLot 데이터셋(12,416장, 약 70만 박스)을 학습하여 mAP50 **0.9944**, 점유율 추정 평균 오차 **0.27%p**를 달성한 end-to-end detection 시스템임.

---

## 왜 이 프로젝트인가

도심 주차난은 차량이 빈 공간을 찾아 헤매며 발생하는 공회전·정체로 이어짐. CCTV 영상에서 주차장 점유 상태를 정확히 파악할 수만 있어도 운전자에게 "어디에 자리가 있는지" 안내할 수 있음. 이 프로젝트는 그 첫 단계인 **시각 인식 모듈**을 끝까지 구현한 결과임.

원래는 [딥러닝 기반 주차 수요 예측 + 자동 배차 시스템(ParkCast)](docs/ParkCast_Proposal.pdf)이라는 더 큰 시스템의 일부로 기획됐으나, 수요 예측·경로 추천 부분은 실제 학습 가능한 데이터가 부족하다고 판단하여 **시각 인식 모듈에 집중**해서 구현함. 시스템 비전은 PDF 보고서로 남겨두고 코어를 완성한 형태.

---

## 결과 요약

### Test set 성능 (PKLot, Roboflow v2 random split)

| Metric | Score |
|---|---|
| mAP50 | **0.9944** |
| mAP50-95 | 0.9886 |
| Precision | 0.9977 |
| Recall | 0.9975 |

### 점유율 추정 정확도 (테스트 200장 샘플)

| Metric | Value |
|---|---|
| 평균 박스 카운트 오차 | 0.03 boxes |
| 평균 점유율 오차 | 0.27 %p |

평균적으로 한 장당 28~70개 칸 중 0.03칸만 틀림. 실용적으로는 거의 정확함.

### 실패 케이스

가장 점유율 추정이 어긋난 사례 6장을 분석한 결과, 공통 원인은 다음과 같았음:
- **트럭/SUV처럼 한 칸을 넘어선 차량**이 두 박스로 잘못 검출됨
- **나무 그림자가 칸을 가리는 경우** 빈 칸을 찬 칸으로 오인
- **원경의 작은 박스**(촬영 시점에서 멀리 있는 칸)에서 누락 발생

자세한 시각화는 `results/failure_cases.png` 참조.

> ⚠️ **Random split의 한계**: PKLot Roboflow v2는 5분 간격 연속 촬영 이미지를 random split했기 때문에, 사실상 거의 동일한 시점의 이미지가 train/test에 섞여 있음. 즉 0.99의 mAP는 *진짜 일반화 성능*이 아님. Week 2에서 이 한계를 검증하고 cross-domain 성능을 따로 측정함 (도메인 갭 분석은 [`docs/week2_cross_domain.md`](docs/week2_cross_domain.md) 참조 — *진행 중*).

---

## 기술 스택

| 분류 | 기술 |
|---|---|
| 모델 | YOLOv8n (Ultralytics) |
| 학습 | PyTorch, transfer learning from COCO |
| 데이터 | PKLot v2 (Roboflow, COCO format → YOLO format 자체 변환) |
| 시각화 | matplotlib, OpenCV |
| 배포 | Gradio (웹 데모) |
| 인프라 | Google Colab (T4 GPU) |

---

## 프로젝트 구조

```
parkcast/
├── parkcast/                     ← import 가능한 라이브러리
│   ├── data.py                   COCO ↔ YOLO 변환, 인덱싱
│   ├── eda.py                    분포·샘플 시각화
│   ├── train.py                  YOLOv8 학습 wrapper
│   ├── inference.py              OccupancyPredictor (단일 이미지 → 점유율)
│   ├── evaluate.py               test mAP + GT 비교 + 실패 케이스 추출
│   ├── visualize.py              박스 그리기, 실패 그리드
│   └── utils.py                  config 로딩, seed, 클래스 헬퍼
│
├── scripts/                      ← 한 줄로 돌릴 수 있는 진입점
│   ├── prepare_data.py           COCO → YOLO 변환
│   ├── run_eda.py                EDA 그림 저장
│   ├── train.py                  학습 실행
│   ├── evaluate.py               평가 + 실패 분석
│   └── predict.py                단일 이미지 추론
│
├── app/
│   └── gradio_demo.py            웹 데모
│
├── configs/
│   └── default.yaml              하이퍼파라미터 한 곳에 모음
│
├── notebooks/                    ← 실험 노트북 (재현용)
│   ├── ParkCast_Week1_YOLOv8.ipynb
│   └── ParkCast_Week2_CrossDomain.ipynb
│
├── docs/
│   └── ParkCast_Proposal.pdf     원래 시스템 비전 (수요 예측 + 배차 추천)
│
├── requirements.txt
└── README.md
```

설계 원칙은 두 개임:
1. **`parkcast/`는 import만으로 재사용 가능**해야 함 (스크립트와 무관하게 동작).
2. **`scripts/`는 한 줄짜리 CLI**로, config 하나만 바꾸면 다른 데이터/하이퍼파라미터에서도 그대로 돌아감.

---

## 빠르게 돌려보기

### 1. 환경 설정

```bash
git clone https://github.com/<your-handle>/parkcast.git
cd parkcast
pip install -r requirements.txt
```

### 2. 데이터 준비

PKLot v2 (Roboflow)를 받아 `/content/`(또는 `configs/default.yaml`의 `raw_root`)에 풀어두면 다음 구조가 됨:

```
raw_root/
├── train/{images, _annotations.coco.json}
├── valid/{images, _annotations.coco.json}
└── test/{images, _annotations.coco.json}
```

### 3. 한 줄씩 실행

```bash
# COCO → YOLO 변환
python scripts/prepare_data.py

# EDA 시각화
python scripts/run_eda.py

# 학습 (T4에서 약 40~60분)
python scripts/train.py

# 평가 + 실패 케이스 추출
python scripts/evaluate.py --weights runs/yolov8n_pklot_v1/weights/best.pt

# 단일 이미지 점유율 예측
python scripts/predict.py --weights runs/yolov8n_pklot_v1/weights/best.pt \
                         --image my_parking_lot.jpg \
                         --save output.png

# 웹 데모 실행
python app/gradio_demo.py --weights runs/yolov8n_pklot_v1/weights/best.pt --share
```

### 4. 라이브러리로 사용

```python
from parkcast.inference import OccupancyPredictor

pred = OccupancyPredictor("models/best.pt")
result = pred.predict("my_lot.jpg", conf=0.4)

print(f"빈 칸: {result.n_empty}")
print(f"찬 칸: {result.n_occupied}")
print(f"점유율: {result.occupancy_pct:.1f}%")
```

---

## 설계 결정 / 트레이드오프

이 섹션이 면접에서 받을 만한 질문에 대한 답임.

### 왜 YOLOv8n인가? (S/M/L 안 쓴 이유)

PKLot의 박스가 한 이미지당 30~70개로 dense하지만 박스 자체는 단순한 직사각형이고 클래스가 2개뿐임. 표현력보다는 **추론 속도**가 더 중요한 응용임 (실시간 CCTV). YOLOv8n으로도 mAP50 0.99가 나오는데 굳이 무거운 모델을 쓸 이유가 없음. 만약 cross-domain 성능이 부족하면 그때 YOLOv8s로 올려서 비교할 계획임.

### 왜 fliplr=0.0 (좌우반전 끔)인가?

PKLot CCTV는 카메라가 고정되어 있어서 좌우반전을 적용하면 학습 분포와 실제 추론 분포가 어긋남. 같은 이유로 강한 회전(`degrees`)도 끔. **데이터의 도메인 특성을 반영한 증강 설계**임.

### 점유율 추정이 0.27%p로 정확한 이유

mAP는 위치+클래스 모두 맞춰야 하지만, 점유율은 **카운트 차이만** 보면 됨. 그래서 박스가 약간 어긋나도(IoU가 낮아도) 카운트는 맞음. mAP보다 더 관대한 metric이라 0.27%p가 실제 응용 정확도에 더 가까움.

### Roboflow random split의 한계 인지

PKLot Roboflow v2는 5분 간격 사진을 random split해서, 사실상 **train과 test가 거의 같은 이미지**가 들어감. mAP 0.99가 그대로 진짜 일반화 성능이 아님. 이 한계를 직접 검증하기 위해 Week 2에서 (1) date split, (2) 이미지 임베딩 클러스터링 기반 cross-lot split 두 가지로 다시 평가함.

---

## 로드맵

- [x] **Week 1**: YOLOv8 베이스라인 학습 + 단일 이미지 점유율 추정
- [ ] **Week 2**: ResNet50 임베딩 + UMAP/K-Means로 도메인 자동 발견 → cross-lot 일반화 평가
- [ ] **Week 3**: 1-stage detection vs 2-stage(차량검출 + 칸 매칭) 비교
- [ ] **Week 4**: Gradio 데모 고도화, 모델 경량화(ONNX/TensorRT) 실험

---

## 데이터셋

**PKLot** (Federal University of Paraná, Brazil)
- 12,416장의 주차장 항공 이미지 (3개 주차장 × 3개 날씨)
- 약 70만 개의 주차칸 어노테이션 (occupied/empty)
- License: CC BY 4.0
- 출처: [Roboflow PKLot v2](https://public.roboflow.ai/object-detection/pklot)

```
Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., Koerich, A.,
PKLot – A robust dataset for parking lot classification,
Expert Systems with Applications, 42(11):4937-4949, 2015.
```

---

## 참고문헌 / 관련 연구

이 프로젝트의 시스템 비전(ParkCast 전체)은 다음 연구를 조합한 것임. 본 레포는 이 중 (1) 시각 인식 모듈을 실제로 구현한 것임.

1. 실내 주차장 차량 위치 추정 시스템 (YOLOv4 + NVDCF Tracker + 다중 호모그래피)
2. CNN + Conv-LSTM + LSTM 기반 시공간 주차 수요 예측
3. Reeds-Shepp Curve + Hybrid A* 기반 주차 경로 생성

자세한 시스템 설계는 [`docs/ParkCast_Proposal.pdf`](docs/ParkCast_Proposal.pdf) 참조.

---

## License

코드: MIT
데이터셋: CC BY 4.0 (PKLot)
