"""Gradio 웹 데모 — 이미지 업로드 → 점유율 추정 + 시각화.

사용:
    python app/gradio_demo.py --weights models/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from parkcast.inference import OccupancyPredictor
from parkcast.utils import YOLO_CLASS_NAMES, is_empty_class


def build_demo(weights: str):
    import gradio as gr

    predictor = OccupancyPredictor(weights, class_names=YOLO_CLASS_NAMES)

    def predict_fn(image: np.ndarray, conf: float):
        # gradio는 RGB로 줌 → 임시 저장 후 추론
        tmp = "/tmp/_parkcast_input.jpg"
        cv2.imwrite(tmp, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        result = predictor.predict(tmp, conf=conf)

        # 박스 그리기
        out = image.copy()
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

    with gr.Blocks(title="ParkCast Vision Demo") as demo:
        gr.Markdown("# ParkCast Vision\n주차장 이미지 → 빈 칸/찬 칸 검출 + 점유율 자동 계산")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--share", action="store_true", help="public link 생성 (Colab용)")
    args = parser.parse_args()

    demo = build_demo(args.weights)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
