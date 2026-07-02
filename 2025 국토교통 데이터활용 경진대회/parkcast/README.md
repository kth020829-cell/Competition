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

> ⚠️ **Random split의 한계**: PKLot Roboflow v2는 5분 간격 연속 촬영 이미지를 random split했기 때문에, 사실상 거의 동일한 시점의 이미지가 train/test에 섞여 있음. 즉 0.99의 mAP는 *진짜 일반화 성능*이 아님. 이 한계를 검증하기 위한 cross-lot 평가 로직은 [`notebooks/ParkCast_Week2_CrossDomain.ipynb`](notebooks/ParkCast_Week2_CrossDomain.ipynb)에서 설계했고 [`parkcast/domain.py`](parkcast/domain.py) + [`scripts/cross_lot_eval.py`](scripts/cross_lot_eval.py)로 재사용 가능하게 정리함 — 코드는 완성됐지만 **실제 학습·평가는 아직 실행 전**(GPU 필요, Colab에서 실행 예정). 실행되면 이 섹션의 mAP 0.9944는 random split 결과라는 전제하에 date/lot split 결과와 나란히 갱신할 것.

---

## Instance Segmentation 결과 (SAM 자동 라벨링 → YOLOv8n-seg)

Week 1(bbox detection) 이후 실제로 Colab에서 돌려서 얻은 첫 번째 고도화 결과. GT bbox를 SAM(Segment Anything)의 box prompt로 넣어 픽셀 단위 마스크를 얻고(`scripts/sam_auto_label.py`), 리소스 제약을 감안해 서브셋(train 2,000 / valid 300 / test 300, `configs/segment.yaml` 참조)으로 YOLOv8n-seg를 학습함.

### 라벨 품질 검증

`parkcast/visualize.py::plot_yolo_seg_label_sample`로 SAM이 뽑은 라벨을 직접 그려본 것 — 각진 bbox가 아니라 차량 윤곽을 따라가는 polygon임을 확인함.

![SAM 자동 라벨링 검증 — 각진 사각형이 아니라 실제 차량 윤곽을 따라감](docs/seg_label_verification.png)

### Test set 성능 (서브셋, box + mask 둘 다 리포트)

| Metric | Score |
|---|---|
| mAP50 (box) | 0.9613 |
| mAP50-95 (box) | 0.7757 |
| Precision | 0.9381 |
| Recall | 0.9335 |
| **Mask mAP50** | **0.9263** |
| **Mask mAP50-95** | **0.5639** |

### 점유율 추정 정확도 (테스트 193장 샘플)

| Metric | Value |
|---|---|
| 평균 박스 카운트 오차 | 1.24 boxes |
| 평균 점유율 오차 | 0.60 %p |

### Detect(bbox) baseline 대비 — 왜 떨어졌는가

| Metric | Detect (전체 8,691장) | Seg (서브셋 2,000장) |
|---|---|---|
| mAP50 | 0.9944 | 0.9613 |
| mAP50-95 | 0.9886 | 0.7757 (box) |
| 평균 점유율 오차 | 0.27 %p | 0.60 %p |

수치만 보면 하락했지만 원인이 명확함: **학습 데이터가 전체의 23%(2,000/8,691장)뿐**이고, seg 모델은 box head와 mask head를 같은 backbone으로 동시에 학습하는 multi-task라 같은 epoch(30)로는 detect 전용 모델만큼 수렴하지 못함. 특히 mAP50-95(높은 IoU 기준까지 정밀하게 맞아야 함)가 mAP50보다 더 크게 떨어진 것도 같은 이유 — "칸을 찾았는지"는 거의 detect만큼 잘하지만 "경계를 얼마나 정밀하게 맞췄는지"는 데이터 부족의 영향을 더 받음. Mask mAP50-95(0.5639)가 box mAP50-95(0.7757)보다 낮은 것도 예상된 결과: `mobile_sam`(가장 가벼운 SAM 변형)의 마스크 정밀도 + `approxPolyDP` 단순화 과정에서 경계 정보 일부 손실. 그래도 점유율 오차 0.60%p는 여전히 실용적 수준임.

**다음 개선 방향**: 서브셋 크기를 늘리거나(`configs/segment.yaml`의 `sam_labeling.subset`), `mobile_sam.pt` 대신 `sam_b.pt`/`sam2_b.pt`로 교체해 마스크 정밀도를 높이는 것으로 mAP50-95 격차를 좁힐 수 있을 것으로 예상함(아직 실험 전).

---

## 기술 스택

| 분류 | 기술 |
|---|---|
| 모델 | YOLOv8n detect / YOLOv8n-seg (Ultralytics) |
| 학습 | PyTorch, transfer learning from COCO |
| 데이터 | PKLot v2 (Roboflow, COCO format → YOLO format/YOLO-seg format 자체 변환) |
| 자동 라벨링 | SAM(mobile_sam/sam_b, Ultralytics 통합) box-prompted segmentation + cv2.findContours/approxPolyDP |
| 도메인 갭 분석 | ResNet50 임베딩 + scikit-learn K-Means/PCA + UMAP |
| VLM 질의 | CLIP (`transformers`, zero-shot 이미지-텍스트 매칭) |
| 시각화 | matplotlib, OpenCV |
| 배포 | Gradio (로컬 웹 데모), HuggingFace Spaces (예정) |
| 인프라 | Google Colab (T4 GPU) |

---

## 프로젝트 구조

```
parkcast/
├── parkcast/                     ← import 가능한 라이브러리
│   ├── data.py                   COCO ↔ YOLO 변환(detect) + COCO ↔ YOLO-seg 변환(segment)
│   ├── eda.py                    분포·샘플 시각화
│   ├── train.py                  YOLOv8 학습 wrapper (detect/seg 공용 — checkpoint로 task 결정)
│   ├── inference.py              OccupancyPredictor (단일 이미지 → 점유율, seg면 마스크도 반환)
│   ├── evaluate.py               test mAP(+seg mAP) + GT 비교 + 실패 케이스 추출
│   ├── visualize.py               박스/폴리곤 그리기, 실패 그리드, YOLO-seg 라벨 검증 시각화
│   ├── domain.py                 cross-lot 도메인 갭 평가 (임베딩 클러스터링, split 구성)
│   ├── sam_label.py               SAM box-prompted 자동 라벨링 (진짜 픽셀 단위 마스크 → polygon)
│   ├── vlm.py                    CLIP 기반 open-vocabulary 질의 (ParkingVLM)
│   └── utils.py                  config 로딩, seed, 클래스 헬퍼
│
├── scripts/                      ← 한 줄로 돌릴 수 있는 진입점
│   ├── prepare_data.py           COCO → YOLO 변환 (config의 task로 detect/segment 분기)
│   ├── run_eda.py                EDA 그림 저장
│   ├── train.py                  학습 실행
│   ├── evaluate.py               평가 + 실패 분석
│   ├── predict.py                단일 이미지 추론
│   ├── cross_lot_eval.py         Random vs Date vs Lot split 비교 (도메인 갭 정량화)
│   └── sam_auto_label.py         SAM으로 YOLO-seg 데이터셋 자동 라벨링 (configs/segment.yaml 사용)
│
├── app/
│   ├── gradio_demo.py            웹 데모 (Detection 탭 + VLM 질의 탭)
│   └── hf_space/                 HuggingFace Spaces 배포용 (app.py, requirements.txt, README.md)
│
├── configs/
│   ├── default.yaml              detect(YOLOv8n) 하이퍼파라미터
│   └── segment.yaml              segment(YOLOv8n-seg) 하이퍼파라미터
│
├── notebooks/                    ← 실험 노트북 (재현용)
│   ├── ParkCast_Week1_YOLOv8.ipynb          실행 완료 (mAP50 0.9944)
│   └── ParkCast_Week2_CrossDomain.ipynb     코드 작성만 완료, 미실행 — scripts/cross_lot_eval.py로 대체
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

### 3. 한 줄씩 실행 (detect, 기존 파이프라인)

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

# 웹 데모 실행 (Detection 탭 + VLM 질의 탭)
python app/gradio_demo.py --weights runs/yolov8n_pklot_v1/weights/best.pt --share
```

### 3-1. Instance Segmentation (YOLOv8n-seg)로 대신 돌리기

라벨 소스는 두 가지 — (A) 빠른 baseline, (B) 실제로 쓰는 SAM 자동 라벨링. 설계는
[`configs/segment.yaml`](configs/segment.yaml) 참조. 둘 다 같은 `yolo_root`에 저장되므로
아래 학습/평가 명령은 방법과 무관하게 동일함:

```bash
# (A) 빠른 baseline — COCO segmentation 필드 그대로/bbox fallback, GPU 불필요
python scripts/prepare_data.py --config configs/segment.yaml

# (B) 실제로 쓰는 방법 — SAM box-prompted 자동 라벨링 (GPU 필요)
python scripts/sam_auto_label.py --config configs/segment.yaml

# 라벨 검증 (SAM 결과가 각진 사각형이 아니라 진짜 윤곽인지 눈으로 확인)
python -c "
from parkcast.visualize import plot_yolo_seg_label_sample
plot_yolo_seg_label_sample('pklot_yolo_seg/train/images/<파일명>.jpg',
                            'pklot_yolo_seg/train/labels/<파일명>.txt',
                            ['spaces', 'space-empty', 'space-occupied'],
                            save_path='seg_label_check.png', show=False)
"

# 학습 + 평가 (box mAP와 mask mAP 둘 다 리포트)
python scripts/train.py    --config configs/segment.yaml
python scripts/evaluate.py --config configs/segment.yaml --weights runs/yolov8n_seg_pklot_v1/weights/best.pt
```

### 3-2. Cross-lot(도메인 갭) 평가

기존 random split 결과를 인자로 넘기면 Date split / Lot split을 새로 만들어 학습·평가하고
셋을 비교함 (`parkcast/domain.py` + `scripts/cross_lot_eval.py`, 아직 실행 전 — GPU 필요):

```bash
python scripts/cross_lot_eval.py --config configs/default.yaml \
    --random-map50 0.9944 --random-map50-95 0.9886 \
    --random-precision 0.9977 --random-recall 0.9975
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

## 고도화 로드맵

Week 1(아래 표) 이후 4가지 방향으로 확장 중. 1단계는 Colab에서 실제로 실행해 결과까지 확보했고, 나머지 3개는 **코드 작성 완료 상태이지 실행/검증까지 끝난 건 아님**(GPU 학습은 Colab에서 순차 진행 예정):

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. Instance Segmentation | YOLOv8n-seg 전환. 라벨은 SAM box-prompted 자동 라벨링(`parkcast/sam_label.py`, GT bbox → SAM → `cv2.findContours`+`approxPolyDP` polygon, 실패 시 bbox fallback)이 실제 방법이고, `coco_to_yolo_seg`(COCO segmentation 필드/bbox fallback)는 GPU 없이 도는 baseline. `inference.py`/`evaluate.py`/`visualize.py`도 마스크 지원 | **✅ 실행 완료** — 서브셋(2,000장) 학습, mAP50 0.9613 / Mask mAP50 0.9263, 상세는 위 "Instance Segmentation 결과" 섹션 참조 |
| 2. HuggingFace Spaces 배포 | `app/hf_space/` (Spaces용 `app.py` + YAML front matter README) | 코드 작성 완료, 실제 push/배포 전 |
| 3. Cross-lot 평가 | `parkcast/domain.py` + `scripts/cross_lot_eval.py` — ResNet50 임베딩 클러스터링으로 주차장 자동 발견 → Random/Date/Lot split 비교 | 코드 작성 완료(Week2 노트북 설계를 모듈화), 실행 전 |
| 4. CLIP 기반 VLM 질의 | `parkcast/vlm.py`(`ParkingVLM`) + `gradio_demo.py`의 "VLM 질의" 탭 — 차종 추정, 자유 텍스트 질의 | 코드 작성 완료, 실행 검증은 로컬 통합 테스트만(transformers 미설치 환경) |

- [x] **Week 1**: YOLOv8 베이스라인 학습 + 단일 이미지 점유율 추정 (실행 완료, mAP50 0.9944)
- [x] **1단계 (Instance Segmentation)**: SAM 자동 라벨링 + YOLOv8n-seg 학습 실행 완료 (mAP50 0.9613 / Mask mAP50 0.9263)
- [ ] **2~4단계**: HuggingFace Spaces 배포 / Cross-lot 평가 / CLIP VLM 질의 — 위 순서대로 Colab에서 실제 실행하고 결과를 이 README/포트폴리오에 반영 예정

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
