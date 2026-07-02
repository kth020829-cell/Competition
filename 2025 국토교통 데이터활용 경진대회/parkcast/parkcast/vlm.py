"""CLIP 기반 open-vocabulary 질의 — YOLO의 고정 2클래스(empty/occupied)를 넘어선 질의.

실패 케이스 분석(README 참조)에서 트럭/SUV처럼 칸 하나를 넘어서는 큰 차량이 두 박스로
오검출되는 경우가 있었음. YOLO는 "비었다/찼다"만 구분하므로 차종·날씨·그림자 같은 정보는
아예 못 줌. CLIP zero-shot 매칭으로 (1) 검출된 박스를 차종 후보군과 비교하거나,
(2) 이미지 전체를 자유 텍스트와 비교해 YOLO가 못 주는 정보를 보완함.

주의: CLIP은 VQA(질의응답) 모델이 아니라 이미지-텍스트 임베딩 유사도 모델임.
"질문에 대답"하는 게 아니라 후보 텍스트들과의 유사도 순위를 매기는 것 — 후보군 설계가
결과 품질을 좌우함. UI에서도 이 한계를 명시함(app/gradio_demo.py).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_VEHICLE_LABELS = [
    "a sedan car",
    "an SUV",
    "a pickup truck or van",
    "a motorcycle",
]

DEFAULT_SCENE_LABELS = [
    "an empty parking lot",
    "a full parking lot",
    "a parking lot with tree shadows on the ground",
    "a snowy parking lot",
    "a rainy parking lot",
    "a parking lot at night",
]


class ParkingVLM:
    """CLIP(openai/clip-vit-base-patch32)을 wrap한 zero-shot 이미지-텍스트 매칭기."""

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def classify(
        self,
        image: np.ndarray,
        candidate_labels: Sequence[str],
        extra_query: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """이미지 전체 vs 후보 텍스트들의 유사도를 softmax 확률로 변환해 내림차순 반환.

        extra_query가 있으면 후보 목록 맨 앞에 추가함(사용자 자유 입력 질의용). 후보가
        전부 비슷하게 안 맞는 이미지라도 그 중 상대적으로 가장 가까운 것이 1등으로 나오므로,
        해석 시 "정답"이 아니라 "후보 중 상대적으로 가장 가까운 것"으로 읽어야 함.
        """
        import torch

        labels = list(candidate_labels)
        if extra_query:
            labels = [extra_query] + labels

        inputs = self.processor(
            text=labels, images=image, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        probs = out.logits_per_image.softmax(dim=1)[0].cpu().numpy()
        return sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])

    def classify_boxes(
        self,
        image: np.ndarray,
        boxes_xyxy: np.ndarray,
        candidate_labels: Sequence[str] = DEFAULT_VEHICLE_LABELS,
    ) -> List[Tuple[str, float]]:
        """YOLO가 검출한 박스마다 crop을 잘라 차종 후보군과 비교 → 박스별 최상위 라벨+확률."""
        results = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = (int(v) for v in box)
            crop = image[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                results.append(("unknown", 0.0))
                continue
            results.append(self.classify(crop, candidate_labels)[0])
        return results
