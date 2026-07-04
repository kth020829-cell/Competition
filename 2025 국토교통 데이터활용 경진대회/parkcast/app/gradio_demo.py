"""Gradio 웹 데모 — 이미지 업로드 → 점유율 추정 + 시각화.

사용:
    python app/gradio_demo.py --weights models/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from parkcast.inference import OccupancyPredictor
from parkcast.utils import YOLO_CLASS_NAMES, is_empty_class


def build_demo(weights: str):
    import gradio as gr

    predictor = OccupancyPredictor(weights, class_names=YOLO_CLASS_NAMES)
    vlm_holder: dict = {}  # CLIP은 무거우므로 VLM 탭을 실제로 쓸 때만 lazy-load

    def get_vlm():
        if "model" not in vlm_holder:
            from parkcast.vlm import ParkingVLM

            vlm_holder["model"] = ParkingVLM()
        return vlm_holder["model"]

    def predict_fn(image: np.ndarray, conf: float):
        # gradio는 RGB로 줌 → 임시 저장 후 추론
        tmp = "/tmp/_parkcast_input.jpg"
        cv2.imwrite(tmp, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        result = predictor.predict(tmp, conf=conf)

        out = image.copy()
        if result.has_masks:
            # seg 모델(yolov8n-seg.pt)이면 polygon을 반투명으로 채워서 칸 형태를 그대로 보여줌
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

    def vlm_query_fn(image: Optional[np.ndarray], query_text: str) -> str:
        if image is None:
            return "이미지를 먼저 업로드하세요."
        from parkcast.vlm import DEFAULT_SCENE_LABELS

        vlm = get_vlm()
        ranked = vlm.classify(image, DEFAULT_SCENE_LABELS, extra_query=query_text or None)
        lines = [f"- **{label}**: {prob * 100:.1f}%" for label, prob in ranked[:5]]
        return "### CLIP 유사도 순위 (상위 5, 후보 간 상대적 비교)\n" + "\n".join(lines)

    def vlm_vehicle_fn(image: Optional[np.ndarray], conf: float) -> str:
        if image is None:
            return "이미지를 먼저 업로드하세요."
        tmp = "/tmp/_parkcast_input.jpg"
        cv2.imwrite(tmp, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        result = predictor.predict(tmp, conf=conf)

        occ_idx = [i for i, n in enumerate(result.class_names) if not is_empty_class(n)]
        if not occ_idx:
            return "검출된 점유 칸이 없습니다."

        vlm = get_vlm()
        if result.has_masks:
            # seg 모델(yolov8n-seg.pt)이면 마스크로 배경을 지운 crop을 씀 — bbox crop보다
            # 옆 칸/배경이 덜 섞이는 게 이론상 장점이지만, Week4 데모(README 참조)에서는
            # 소규모 샘플로 이 이점이 뚜렷하게 확인되진 않았음
            masks = [result.masks_xy[i] for i in occ_idx]
            ranked = vlm.classify_masks(image, masks)
            method_note = "YOLO 마스크(배경 제거) + CLIP zero-shot"
        else:
            boxes = result.boxes_xyxy[occ_idx]
            ranked = vlm.classify_boxes(image, boxes)
            method_note = "YOLO 박스 + CLIP zero-shot"

        lines = [f"- Box {i + 1}: **{label}** ({prob * 100:.1f}%)" for i, (label, prob) in enumerate(ranked)]
        return f"### 점유 칸별 차종 추정 ({method_note})\n" + "\n".join(lines)

    with gr.Blocks(title="ParkCast Vision Demo") as demo:
        gr.Markdown("# ParkCast Vision\n주차장 이미지 → 빈 칸/찬 칸 검출 + 점유율 자동 계산")
        gr.Markdown(
            "> ⚠️ **PKLot 데이터셋(브라질 대학 주차장을 정면 위에서 내려다본 항공사진)으로만 "
            "학습됨.** 지도/위성 스크린샷이나 도로가 함께 보이는 비스듬한 각도의 사진은 학습 "
            "도메인과 많이 달라 검출이 거의 안 될 수 있음(버그가 아니라 도메인 갭임 — README의 "
            "\"Cross-lot 도메인 갭 평가\" 섹션 참조). 정면에 가깝게, 주차칸이 규칙적으로 보이는 "
            "항공사진일수록 잘 됨."
        )
        with gr.Tabs():
            with gr.Tab("Detection"):
                with gr.Row():
                    with gr.Column():
                        inp = gr.Image(type="numpy", label="Parking lot image")
                        conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Confidence threshold")
                        btn = gr.Button("Detect", variant="primary")
                    with gr.Column():
                        out_img = gr.Image(label="Detection")
                        out_md = gr.Markdown()
                btn.click(predict_fn, [inp, conf], [out_img, out_md])

            with gr.Tab("VLM 질의 (CLIP)"):
                gr.Markdown(
                    "YOLO는 학습된 empty/occupied 2클래스만 구분하지만, CLIP은 학습되지 않은 "
                    "개념도 텍스트로 비교 가능함(차종·날씨·그림자 등). **주의**: CLIP은 질문에 "
                    "답하는 VQA 모델이 아니라 이미지-텍스트 임베딩 유사도 모델임 — 아래 결과는 "
                    "후보 문장들 사이의 상대적 순위이지, 정답을 보장하지 않음."
                )
                with gr.Row():
                    with gr.Column():
                        vlm_inp = gr.Image(type="numpy", label="Parking lot image")
                        vlm_query = gr.Textbox(
                            label="자유 질의 (영어 권장, 예: 'a truck blocking two spaces')",
                            placeholder="비워두면 기본 후보 목록만 비교",
                        )
                        vlm_scene_btn = gr.Button("장면 질의 실행")
                        vlm_vehicle_btn = gr.Button("점유 칸별 차종 추정 (YOLO+CLIP)")
                    with gr.Column():
                        vlm_out = gr.Markdown()
                vlm_scene_btn.click(vlm_query_fn, [vlm_inp, vlm_query], vlm_out)
                vlm_vehicle_btn.click(vlm_vehicle_fn, [vlm_inp, conf], vlm_out)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--share", action="store_true", help="공개 접속 링크 생성")
    args = parser.parse_args()

    demo = build_demo(args.weights)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
