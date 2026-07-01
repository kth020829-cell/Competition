from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal
import asyncio
from datetime import datetime, timezone
from boto3.dynamodb.conditions import Key

from app.services.feature_engineering import (
    UserProfile, ClimateFeatures, BioFeatures,
)
from app.services.chai_scorer import compute_chai
from app.services.data_collector import KMAClient, AirKoreaClient, parse_forecast, parse_cai
from app.models.autoencoder import detect_anomaly
from app.services.synthetic_generator import WearableSimulator
from app.db import dynamodb

router = APIRouter(prefix="/predictions", tags=["predictions"])


class BioInput(BaseModel):
    hr_bpm: float = Field(gt=0, lt=300)
    hrv_ms: float = Field(ge=0)
    skin_temp_c: float = Field(ge=30, le=42)
    spo2_pct: float = Field(ge=50, le=100)


class PredictionRequest(BaseModel):
    user_id: str
    bio: BioInput
    nx: int = 60   # 기상청 격자 X (기본 서울 중구)
    ny: int = 127  # 기상청 격자 Y
    station_name: str = "중구"
    use_shap: bool = False


@router.post("/chai")
async def predict_chai(req: PredictionRequest):
    # 사용자 프로필 조회
    user_item = dynamodb.get_item("aider-users", {"user_id": req.user_id})
    if not user_item:
        raise HTTPException(status_code=404, detail="User not found")

    profile = UserProfile(
        user_id=user_item["user_id"],
        age=int(user_item["age"]),
        gender=user_item["gender"],
        region_code=user_item.get("region_code", "11"),
        has_hypertension=bool(user_item.get("has_hypertension", False)),
        has_diabetes=bool(user_item.get("has_diabetes", False)),
        has_heart_disease=bool(user_item.get("has_heart_disease", False)),
        has_respiratory=bool(user_item.get("has_respiratory", False)),
    )

    # 외부 API 병렬 호출
    kma = KMAClient()
    air = AirKoreaClient()
    forecast_raw, cai_raw = await asyncio.gather(
        kma.get_vilage_forecast(nx=req.nx, ny=req.ny),
        air.get_cai(station_name=req.station_name),
    )
    fc = parse_forecast(forecast_raw)
    cai = parse_cai(cai_raw)

    climate = ClimateFeatures(
        temp_c=fc["temp_c"] or 25.0,
        humidity_pct=fc["humidity_pct"] or 60.0,
        wind_speed_ms=fc["wind_speed_ms"] or 2.0,
        feels_like_c=fc["feels_like_c"] or 25.0,
        pm25=cai["pm25"] or 15.0,
        pm10=cai["pm10"] or 30.0,
        cai_index=cai["cai_index"] or 50.0,
    )

    bio = BioFeatures(
        hr_bpm=req.bio.hr_bpm,
        hrv_ms=req.bio.hrv_ms,
        skin_temp_c=req.bio.skin_temp_c,
        spo2_pct=req.bio.spo2_pct,
    )

    # 과거 Et 히스토리 (최근 10개)
    past_records = dynamodb.query_items(
        "aider-records",
        Key("user_id").eq(req.user_id),
        Limit=10,
        ScanIndexForward=False,
    )
    historical_et = [float(r["features"]["et"]) for r in past_records if "features" in r and "et" in r["features"]]

    chai_result = compute_chai(profile, climate, bio, historical_et, use_shap=req.use_shap)

    # 기록 저장
    record = {
        "user_id": req.user_id,
        "timestamp": chai_result["timestamp"],
        "chai_score": str(chai_result["score"]),
        "risk_level": chai_result["risk_level"],
        "features": {k: str(v) for k, v in chai_result["features"].items()},
    }
    dynamodb.put_item("aider-records", record)

    return {
        **chai_result,
        "climate_data": {**fc, **cai},
        "data_source": {"forecast": "mock" if forecast_raw.get("_mock") else "kma", "air": "mock" if cai_raw.get("_mock") else "airkorea"},
    }


@router.get("/simulate")
def simulate_scenario(
    stress_type: Literal["none", "heat", "cold"] = Query("heat"),
    outdoor_temp: float = Query(35.0),
    duration_hours: int = Query(6, ge=1, le=24),
):
    """합성 시나리오 생성 + CHAI 계산 (데모용)."""
    sim = WearableSimulator()
    df = sim.generate(stress_type=stress_type, duration_min=duration_hours * 60, seed=42)
    last = df.iloc[-1]

    bio = BioFeatures(
        hr_bpm=float(last["hr_bpm"]),
        hrv_ms=float(last["hrv_ms"]),
        skin_temp_c=float(last["skin_temp_c"]),
        spo2_pct=float(last["spo2_pct"]),
    )

    temp_map = {"heat": outdoor_temp, "cold": max(-10, outdoor_temp - 40), "none": 22.0}
    climate = ClimateFeatures(
        temp_c=temp_map[stress_type],
        humidity_pct=65.0,
        wind_speed_ms=2.0,
        feels_like_c=temp_map[stress_type] - 1.0,
        pm25=18.0,
        pm10=35.0,
        cai_index=75.0,
        heat_warning_level=1 if stress_type == "heat" else 0,
        cold_warning_level=1 if stress_type == "cold" else 0,
    )

    profile = UserProfile(
        user_id="demo",
        age=72,
        gender="M",
        has_hypertension=True,
        region_code="11110",
    )

    anomaly = detect_anomaly(df)
    chai_result = compute_chai(profile, climate, bio)

    return {
        **chai_result,
        "anomaly_detection": anomaly,
        "simulated_bio": {
            "hr_bpm": float(last["hr_bpm"]),
            "hrv_ms": float(last["hrv_ms"]),
            "skin_temp_c": float(last["skin_temp_c"]),
            "spo2_pct": float(last["spo2_pct"]),
        },
        "stress_type": stress_type,
        "simulation_mode": True,
    }


@router.get("/history/{user_id}")
def get_history(user_id: str, limit: int = Query(20, ge=1, le=100)):
    records = dynamodb.query_items(
        "aider-records",
        Key("user_id").eq(user_id),
        Limit=limit,
        ScanIndexForward=False,
    )
    return {"user_id": user_id, "records": records}
