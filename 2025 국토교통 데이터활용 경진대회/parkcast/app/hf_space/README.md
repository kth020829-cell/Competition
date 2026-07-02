---
title: ParkCast Vision
emoji: 🅿️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# ParkCast Vision — HF Spaces 배포본

`app.py`는 [메인 프로젝트](../../README.md)의 `app/gradio_demo.py`와 로직은 동일하고,
가중치를 로컬 `--weights` 인자 대신 자동으로 찾는다는 점만 다름 (`app.py`의 `_resolve_weights` 참조).

CLIP 기반 VLM 질의 탭(`app/gradio_demo.py`에만 있음)은 이 Space 배포본에는 포함하지 않음 —
CLIP 가중치(~600MB)까지 얹으면 Space 콜드 스타트가 크게 느려지기 때문. VLM 탭까지 필요하면
`app.py`를 `app/gradio_demo.py` 기준으로 다시 맞추고 `requirements.txt`에 `transformers`를
추가할 것.

## 배포 절차 (수동)

이 Space는 아직 실제로 push된 적 없음 — 아래는 배포 시 따라야 할 절차:

1. HF Hub에 모델 저장소 생성 후 학습된 `best.pt` 업로드 (예: `<username>/parkcast-yolov8n`)
2. `huggingface-cli repo create parkcast-vision --type space --sdk gradio` (또는 웹 UI로 Space 생성)
3. 이 폴더(`app/hf_space/`) 내용 전체 + 프로젝트 루트의 `parkcast/` 패키지 폴더를 Space repo 루트에 복사
   ```
   space-repo/
   ├── app.py
   ├── requirements.txt
   ├── README.md
   └── parkcast/          ← 프로젝트 루트에서 복사
   ```
4. Space Settings → Repository secrets/variables에 `PARKCAST_HF_MODEL_REPO=<username>/parkcast-yolov8n` 등록
   (또는 `models/best.pt`를 Space repo에 직접 포함 — YOLOv8n은 수 MB 수준이라 git으로도 무리 없음)
5. `git push`로 Space에 반영 → 자동 빌드

## 로컬에서 미리 확인

```bash
cd parkcast
python app/hf_space/app.py
```

프로젝트 루트의 `parkcast/`를 그대로 import하므로 Space에 올리기 전 로컬 검증 가능.
단, 가중치가 없으면 `_resolve_weights()`가 `FileNotFoundError`를 던짐 — 이 폴더(`app/hf_space/`)
아래에 `models/best.pt`를 두거나 `PARKCAST_HF_MODEL_REPO` 환경변수를 설정할 것.
