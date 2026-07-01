"""LightGBM CHAI 스코어 예측 모델 래퍼."""
import os
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from app.config import get_settings

_lgbm_model = None
_model_path = None


def _get_model_path() -> Path:
    return Path(get_settings().model_artifact_dir) / "lgbm_chai.pkl"


def load_model() -> lgb.Booster | None:
    global _lgbm_model, _model_path
    path = _get_model_path()
    if path.exists():
        _lgbm_model = joblib.load(path)
        _model_path = path
    return _lgbm_model


def get_model() -> lgb.Booster | None:
    global _lgbm_model
    if _lgbm_model is None:
        load_model()
    return _lgbm_model


FEATURE_NAMES = ["ct", "bt", "et", "v_personal", "st", "rt"]


def _to_df(features: np.ndarray):
    import pandas as pd
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return pd.DataFrame(features, columns=FEATURE_NAMES)


def predict(features: np.ndarray) -> float:
    """
    features: shape (6,) — [ct, bt, et, v_personal, st, rt]
    반환: CHAI 스코어 ∈ [0, 1]
    """
    model = get_model()
    if model is None:
        return _rule_based_fallback(features)

    score = float(model.predict(_to_df(features))[0])
    return float(np.clip(score, 0.0, 1.0))


def predict_with_shap(features: np.ndarray) -> dict:
    """SHAP 설명값 포함 예측."""
    import shap
    model = get_model()
    if model is None:
        score = _rule_based_fallback(features)
        return {"score": score, "shap_values": None}

    df = _to_df(features)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(df)

    score = float(np.clip(model.predict(df)[0], 0.0, 1.0))
    shap_dict = {name: float(val) for name, val in zip(FEATURE_NAMES, shap_vals[0])}
    return {"score": score, "shap_values": shap_dict}


def _rule_based_fallback(features: np.ndarray) -> float:
    """모델 아티팩트 없을 때 수식 기반 폴백."""
    ct, bt, et, v_personal, st, rt = features.flat
    settings = get_settings()
    w_personal = settings.alpha * v_personal + settings.beta * st + settings.gamma * rt
    chai = float(np.clip(0.4 * ct + 0.3 * bt + 0.2 * et + 0.1 * w_personal, 0.0, 1.0))
    return chai


def save_model(model: lgb.Booster):
    path = _get_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    global _lgbm_model
    _lgbm_model = model
