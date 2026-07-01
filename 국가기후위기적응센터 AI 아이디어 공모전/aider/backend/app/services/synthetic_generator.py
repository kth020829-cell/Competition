"""
합성 데이터 생성기
- WESAD/PPG-DaLiA 분포 참고: 폭염/한파 스트레스에 따른 HRV 저하·체온 상승 패턴
- 실내 열전달 RC 모델: 외부 기온 → 실내 온습도 시차 반영
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Literal


class WearableSimulator:
    """폭염·한파 스트레스에 따른 생체신호 시계열 합성."""

    # 정상 범위 (참고: WESAD 기반)
    BASE_HR = 72.0       # bpm
    BASE_HRV = 55.0      # ms (RMSSD)
    BASE_TEMP = 36.6     # °C (피부 체온)
    BASE_SPO2 = 98.0     # %

    def generate(
        self,
        stress_type: Literal["none", "heat", "cold"],
        duration_min: int = 60,
        sample_rate_sec: int = 60,
        seed: int | None = None,
    ) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)

        n = duration_min * 60 // sample_rate_sec
        t = np.arange(n)

        # 스트레스 계수
        stress_map = {"none": 0.0, "heat": 1.0, "cold": -1.0}
        s = stress_map[stress_type]

        # HR: 폭염 시 상승, 한파 시 약간 상승 (혈관 수축)
        hr_delta = s * 12 + abs(s) * 5
        hr = self.BASE_HR + hr_delta + np.random.normal(0, 2.5, n)
        hr += 3 * np.sin(2 * np.pi * t / (n * 0.3))  # 서카디안 미사용, 단기 진동

        # HRV: 스트레스 시 감소
        hrv_delta = -abs(s) * 18
        hrv = np.maximum(5, self.BASE_HRV + hrv_delta + np.random.normal(0, 4, n))

        # 피부 체온: 폭염 시 상승, 한파 시 하강
        temp_delta = s * 0.8
        skin_temp = self.BASE_TEMP + temp_delta + np.random.normal(0, 0.15, n)

        # SpO2: 극단적 스트레스 시 약간 감소
        spo2 = self.BASE_SPO2 - abs(s) * 0.5 + np.random.normal(0, 0.2, n)
        spo2 = np.clip(spo2, 90, 100)

        now = datetime.now(timezone.utc)
        timestamps = [now - timedelta(seconds=(n - i) * sample_rate_sec) for i in range(n)]

        return pd.DataFrame({
            "timestamp": timestamps,
            "stress_type": stress_type,
            "hr_bpm": hr.round(1),
            "hrv_ms": hrv.round(1),
            "skin_temp_c": skin_temp.round(2),
            "spo2_pct": spo2.round(1),
        })


class IndoorEnvironmentSimulator:
    """
    RC 열전달 모델: τ·dT_in/dt = T_out - T_in
    τ = RC 시상수 (기본 2시간 = 주거 건물 단열 수준)
    """

    def __init__(self, tau_hours: float = 2.0):
        self.tau = tau_hours * 3600  # seconds

    def simulate(
        self,
        outdoor_temps: list[float],
        sample_rate_sec: int = 600,
        initial_indoor: float | None = None,
    ) -> pd.DataFrame:
        n = len(outdoor_temps)
        T_out = np.array(outdoor_temps)

        T_in = np.zeros(n)
        T_in[0] = initial_indoor if initial_indoor is not None else T_out[0]
        dt = sample_rate_sec
        alpha = dt / self.tau

        for i in range(1, n):
            T_in[i] = T_in[i - 1] + alpha * (T_out[i - 1] - T_in[i - 1])

        # 실내 습도: 외기 습도에 지연 + 에어컨 효과(여름: 강제 55% RH 보정)
        base_rh = 60.0
        indoor_rh = base_rh + 0.3 * (T_out - T_out.mean()) + np.random.normal(0, 2, n)
        indoor_rh = np.clip(indoor_rh, 20, 90)

        now = datetime.now(timezone.utc)
        timestamps = [now - timedelta(seconds=(n - i) * sample_rate_sec) for i in range(n)]

        return pd.DataFrame({
            "timestamp": timestamps,
            "outdoor_temp_c": T_out.round(2),
            "indoor_temp_c": T_in.round(2),
            "indoor_rh_pct": indoor_rh.round(1),
            "temp_lag_delta": (T_in - T_out).round(2),
        })


def generate_full_scenario(
    stress_type: Literal["none", "heat", "cold"] = "heat",
    outdoor_temp_base: float = 35.0,
    duration_hours: int = 6,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """웨어러블 + 실내 환경 합성 데이터 한 번에 생성."""
    wearable = WearableSimulator().generate(
        stress_type=stress_type,
        duration_min=duration_hours * 60,
        seed=seed,
    )

    n_env = duration_hours * 6  # 10분 간격
    if stress_type == "heat":
        outdoor_temps = [outdoor_temp_base + 2 * np.sin(2 * np.pi * i / n_env) for i in range(n_env)]
    elif stress_type == "cold":
        outdoor_temps = [outdoor_temp_base - 3 * abs(np.sin(2 * np.pi * i / n_env)) for i in range(n_env)]
    else:
        outdoor_temps = [outdoor_temp_base + np.random.normal(0, 1) for _ in range(n_env)]

    indoor = IndoorEnvironmentSimulator().simulate(outdoor_temps=outdoor_temps, sample_rate_sec=600)

    return {"wearable": wearable, "indoor": indoor}
