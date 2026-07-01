"""
CHAI 스코어 통합 계산기
CHAI_t = σ(f(Ct, Bt, Et) + w_personal)
w_personal = α·V_personal + β·St + γ·Rt
"""
from datetime import datetime, timezone
from app.config import get_settings
from app.services.feature_engineering import (
    UserProfile, ClimateFeatures, BioFeatures,
    build_feature_vector, CHAIFeatureVector,
)
from app.models import lgbm_model


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "critical"
    elif score >= 0.55:
        return "high"
    elif score >= 0.35:
        return "moderate"
    else:
        return "low"


def risk_label_ko(score: float) -> str:
    level = risk_level(score)
    return {"critical": "위험", "high": "높음", "moderate": "보통", "low": "낮음"}[level]


def compute_chai(
    profile: UserProfile,
    climate: ClimateFeatures,
    bio: BioFeatures,
    historical_et: list[float] | None = None,
    use_shap: bool = False,
) -> dict:
    """
    CHAI 점수 계산 메인 함수.
    Returns: {score, risk_level, risk_label_ko, features, shap_values?, timestamp}
    """
    fv: CHAIFeatureVector = build_feature_vector(profile, climate, bio, historical_et)
    arr = fv.to_array()

    if use_shap:
        result = lgbm_model.predict_with_shap(arr)
        score = result["score"]
        shap_values = result["shap_values"]
    else:
        score = lgbm_model.predict(arr)
        shap_values = None

    output = {
        "score": round(score, 4),
        "risk_level": risk_level(score),
        "risk_label_ko": risk_label_ko(score),
        "features": fv.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if shap_values is not None:
        output["shap_values"] = shap_values

    return output
