"""
LightGBM CHAI 스코어 학습 스크립트
합성 데이터 기반으로 [Ct, Bt, Et, V_personal, St, Rt] → CHAI 타겟 학습
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path

from app.services.synthetic_generator import WearableSimulator
from app.services.feature_engineering import (
    UserProfile, ClimateFeatures, BioFeatures,
    build_feature_vector, compute_ct, compute_bt, compute_et,
    compute_v_personal,
)
from app.models.lgbm_model import save_model

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

SEED = 42
N_SAMPLES = 5000


def generate_training_data(n: int = N_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sim = WearableSimulator()
    rows = []

    stress_types = ["none", "heat", "cold"]
    for i in range(n):
        stress = rng.choice(stress_types)
        df_w = sim.generate(stress_type=stress, duration_min=60, seed=int(rng.integers(0, 99999)))
        last = df_w.iloc[-1]

        bio = BioFeatures(
            hr_bpm=float(last["hr_bpm"]),
            hrv_ms=float(last["hrv_ms"]),
            skin_temp_c=float(last["skin_temp_c"]),
            spo2_pct=float(last["spo2_pct"]),
        )

        temp_base = {"none": rng.uniform(15, 28), "heat": rng.uniform(30, 40), "cold": rng.uniform(-10, 5)}[stress]
        climate = ClimateFeatures(
            temp_c=float(temp_base),
            humidity_pct=float(rng.uniform(40, 85)),
            wind_speed_ms=float(rng.uniform(0.5, 7)),
            feels_like_c=float(temp_base - rng.uniform(0, 3)),
            pm25=float(rng.uniform(5, 80)),
            pm10=float(rng.uniform(10, 150)),
            cai_index=float(rng.uniform(20, 200)),
            heat_warning_level=int(rng.integers(0, 3)) if stress == "heat" else 0,
            cold_warning_level=int(rng.integers(0, 3)) if stress == "cold" else 0,
        )

        age = int(rng.integers(45, 90))
        profile = UserProfile(
            user_id=f"synth_{i}",
            age=age,
            gender=rng.choice(["M", "F"]),
            has_hypertension=bool(rng.random() < 0.3),
            has_diabetes=bool(rng.random() < 0.2),
            has_heart_disease=bool(rng.random() < 0.15),
            has_respiratory=bool(rng.random() < 0.1),
            region_code=rng.choice(["11110", "11140", "11170", "11200", "11215"]),
        )

        fv = build_feature_vector(profile, climate, bio, historical_et=None)

        # 합성 타겟: 수식 기반 + 노이즈 (규칙 기반 참값 시뮬레이션)
        w_personal = 0.5 * fv.v_personal + 0.3 * fv.st + 0.2 * fv.rt
        target = np.clip(
            0.4 * fv.ct + 0.3 * fv.bt + 0.2 * fv.et + 0.1 * w_personal
            + rng.normal(0, 0.02),
            0.0, 1.0
        )

        rows.append({
            "ct": fv.ct, "bt": fv.bt, "et": fv.et,
            "v_personal": fv.v_personal, "st": fv.st, "rt": fv.rt,
            "chai_score": float(target),
        })

    return pd.DataFrame(rows)


def train():
    print("데이터 생성 중...")
    df = generate_training_data(N_SAMPLES)
    print(f"샘플 수: {len(df)}, CHAI 평균: {df['chai_score'].mean():.3f}")

    feature_cols = ["ct", "bt", "et", "v_personal", "st", "rt"]
    X = df[feature_cols].values
    y = df["chai_score"].values

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 300,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": SEED,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"Validation MAE: {mae:.4f}")
    print("피처 중요도:", dict(zip(feature_cols, model.feature_importances_)))

    save_model(model.booster_)
    print(f"모델 저장 완료: {ARTIFACT_DIR / 'lgbm_chai.pkl'}")


if __name__ == "__main__":
    train()
