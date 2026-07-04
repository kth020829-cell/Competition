"""HuggingFace Spaces 진입점.

app/gradio_demo.py(로컬용, --weights 인자 필수)와 다른 점은 가중치 소스뿐임:
1) 이 폴더에 함께 올린 models/best.pt가 있으면 그걸 쓰고,
2) 없으면 환경변수 PARKCAST_HF_MODEL_REPO(HF Hub 모델 저장소 id)에서 자동 다운로드함.

Spaces에 올릴 때 필요한 것 (README.md의 "HF Spaces 배포" 절 참조):
  - 이 파일(app.py) + requirements.txt + README.md(YAML front matter 포함)
  - parkcast/ 패키지 폴더 전체를 이 디렉토리 옆에 복사
  - (선택) models/best.pt를 이 디렉토리에 포함 — 없으면 PARKCAST_HF_MODEL_REPO 필요
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))                 # Space 배포본: 옆에 복사된 parkcast/
sys.path.insert(0, str(_THIS_DIR.parent.parent))    # 로컬 개발: 프로젝트 루트의 parkcast/

import cv2
import gradio as gr
import numpy as np

from parkcast.inference import OccupancyPredictor
from parkcast.utils import YOLO_CLASS_NAMES, is_empty_class


def _resolve_weights() -> str:
    """가중치 경로 결정: 번들된 models/best.pt → HF Hub 다운로드 순."""
    local = _THIS_DIR / "models" / "best.pt"
    if local.exists():
        return str(local)

    repo_id = os.environ.get("PARKCAST_HF_MODEL_REPO")
    if repo_id:
        from huggingface_hub import hf_hub_download

        filename = os.environ.get("PARKCAST_HF_MODEL_FILE", "best.pt")
        return hf_hub_download(repo_id=repo_id, filename=filename)

    raise FileNotFoundError(
        "가중치를 찾을 수 없음. models/best.pt를 이 디렉토리에 포함하거나, "
        "Space Settings > Variables에서 PARKCAST_HF_MODEL_REPO(HF Hub 모델 저장소 id)를 설정하세요."
    )


def build_demo() -> gr.Blocks:
    predictor = OccupancyPredictor(_resolve_weights(), class_names=YOLO_CLASS_NAMES)

    def predict_fn(image: np.ndarray, conf: float):
        tmp = "/tmp/_parkcast_input.jpg"
        cv2.imwrite(tmp, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        result = predictor.predict(tmp, conf=conf)

        out = image.copy()
        if result.has_masks:
            overlay = out.copy()
            for poly, name in zip(result.masks_xy, result.class_names):
                color = (0, 255, 0) if is_empty_class(name) else (255, 0, 0)
                cv2.fillPoly(overlay, [poly.astype(np.int32)], color)
            out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)

        for (x1, y1, x2, y2), name in zip(result.boxes_xyxy, result.class_names):
            color = (0, 255, 0) if is_empty_class(name) else (255, 0, 0)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        text = (
            f"### Result\n"
            f"- **Empty:** {result.n_empty}\n"
            f"- **Occupied:** {result.n_occupied}\n"
            f"- **Total:** {result.n_total}\n"
            f"- **Occupancy rate:** {result.occupancy_pct:.1f}%"
        )
        return out, text

    with gr.Blocks(title="ParkCast Vision") as demo:
        gr.Markdown(
            "# ParkCast Vision\n"
            "주차장 이미지 한 장 → 빈 칸/찬 칸 검출 + 점유율 자동 계산.\n"
            "PKLot 데이터셋(YOLOv8n)으로 학습됨 — mAP50 0.9944 (random split, 상세는 README 참조)."
        )
        gr.Markdown(
            "> ⚠️ **PKLot 데이터셋(브라질 대학 주차장을 정면 위에서 내려다본 항공사진)으로만 "
            "학습됨.** 지도/위성 스크린샷이나 도로가 함께 보이는 비스듬한 각도의 사진은 학습 "
            "도메인과 많이 달라 검출이 거의 안 될 수 있음(모델 버그가 아니라 도메인 갭임). "
            "정면에 가깝게, 주차칸이 규칙적으로 보이는 항공사진일수록 잘 됨."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="Parking lot image")
                conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Confidence threshold")
                btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Detection")
                out_md = gr.Markdown()

        btn.click(predict_fn, [inp, conf], [out_img, out_md])

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
