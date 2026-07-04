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

## 배포 절차

Space는 이미 생성돼 있고(SDK가 **Gradio**인지 Space Settings에서 먼저 확인할 것),
가중치는 Colab의 Drive에 있으므로 로컬 git clone/push보다 Colab에서 `huggingface_hub`로
직접 업로드하는 게 더 간단함 — `notebooks/ParkCast_Week4_VLM.ipynb`의 "6. HuggingFace
Spaces 배포" 셀 참조. 요약하면:

```python
from huggingface_hub import login, HfApi
login()  # write 권한 토큰 필요 (huggingface.co/settings/tokens)

api = HfApi()
api.upload_folder(folder_path=".../app/hf_space", repo_id="<username>/<space-name>", repo_type="space")
api.upload_folder(folder_path=".../parkcast", repo_id="<username>/<space-name>", repo_type="space", path_in_repo="parkcast")
api.upload_file(path_or_fileobj="<가중치 경로>", repo_id="<username>/<space-name>", repo_type="space", path_in_repo="models/best.pt")
```

git 기반으로 하고 싶으면 대신 이렇게:

1. `git clone https://huggingface.co/spaces/<username>/<space-name>`
2. 이 폴더(`app/hf_space/`) 내용 전체 + 프로젝트 루트의 `parkcast/` 패키지 폴더를 그 clone 루트에 복사
   ```
   space-repo/
   ├── app.py
   ├── requirements.txt
   ├── README.md
   ├── parkcast/          ← 프로젝트 루트에서 복사
   └── models/best.pt     ← 학습된 가중치
   ```
3. `git add . && git commit -m "deploy" && git push` (HF 토큰을 비밀번호로 사용)

가중치를 Space repo에 직접 안 넣고 별도 HF Hub 모델 저장소에서 받고 싶으면, Space Settings →
Variables에 `PARKCAST_HF_MODEL_REPO=<username>/<model-repo>`를 등록하면 `_resolve_weights()`가
자동으로 `hf_hub_download`로 받아옴.

## 로컬에서 미리 확인

```bash
cd parkcast
python app/hf_space/app.py
```

프로젝트 루트의 `parkcast/`를 그대로 import하므로 Space에 올리기 전 로컬 검증 가능.
단, 가중치가 없으면 `_resolve_weights()`가 `FileNotFoundError`를 던짐 — 이 폴더(`app/hf_space/`)
아래에 `models/best.pt`를 두거나 `PARKCAST_HF_MODEL_REPO` 환경변수를 설정할 것.
