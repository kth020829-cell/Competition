"""
CHAI 모델 피처 엔지니어링
- Ct: 기후 스트레스 지수 (기상청 API 기반)
- Bt: 생체 반응 점수 (Transformer 인코더 출력 또는 통계 기반 대체)
- Et: 환경 노출 지수 (EMA 방식)
- V_personal: 개인 취약성 (건보공단 통계 기반 고정 가중치)
- St: 사회 취약성 (경로당 현황 등 지역 집계)
- Rt: 회복력 (의료 접근성 등)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class UserProfile:
    user_id: str
    age: int
    gender: str          # M / F
    has_hypertension: bool = False
    has_diabetes: bool = False
    has_heart_disease: bool = False
    has_respiratory: bool = False
    region_code: str = "11"  # 기본 서울


@dataclass
class ClimateFeatures:
    temp_c: float
    humidity_pct: float
    wind_speed_ms: float
    feels_like_c: float
    pm25: float
    pm10: float
    cai_index: float
    heat_warning_level: int = 0  # 0=없음, 1=주의보, 2=경보
    cold_warning_level: int = 0


@dataclass
class BioFeatures:
    hr_bpm: float
    hrv_ms: float
    skin_temp_c: float
    spo2_pct: float


def compute_ct(climate: ClimateFeatures) -> float:
    """
    기후 스트레스 지수 Ct ∈ [0, 1]
    - 체감온도 극단값 + 공기질 + 특보 가중합
    """
    # 폭염 스트레스 (33°C 이상부터 선형 증가)
    heat_stress = max(0.0, (climate.feels_like_c - 33) / 10)

    # 한파 스트레스 (0°C 이하부터 선형 증가)
    cold_stress = max(0.0, (-climate.feels_like_c) / 15)

    # 공기질 기여 (CAI 0~500 → 0~0.4 정규화)
    air_stress = min(0.4, climate.cai_index / 500 * 0.4)

    # 특보 가산점
    warning_bonus = climate.heat_warning_level * 0.2 + climate.cold_warning_level * 0.15

    ct = heat_stress + cold_stress + air_stress + warning_bonus
    return float(np.clip(ct, 0.0, 1.0))


def compute_bt(bio: BioFeatures) -> float:
    """
    생체 반응 점수 Bt ∈ [0, 1] — 단순 통계 기반 (Transformer 없을 때 폴백).
    정상 범위에서 멀수록 높아짐.
    """
    # HR 이탈 (정상 60~100 bpm)
    hr_dev = max(0.0, abs(bio.hr_bpm - 75) - 15) / 40

    # HRV 이탈 (높을수록 좋음 — 40ms 이하면 스트레스)
    hrv_score = max(0.0, (40 - bio.hrv_ms) / 35)

    # 체온 이탈 (정상 36.1~37.2)
    temp_dev = max(0.0, abs(bio.skin_temp_c - 36.6) - 0.5) / 1.5

    # SpO2 이탈 (95 이하면 위험)
    spo2_score = max(0.0, (95 - bio.spo2_pct) / 10)

    bt = 0.3 * hr_dev + 0.35 * hrv_score + 0.2 * temp_dev + 0.15 * spo2_score
    return float(np.clip(bt, 0.0, 1.0))


def compute_et(
    historical_et: list[float],
    current_ct: float,
    alpha: float = 0.3,
) -> float:
    """
    환경 노출 지수 Et: EMA(과거 노출 이력 + 현재 Ct)
    α=0.3: 현재 값 가중치, 과거 누적 강조
    """
    if not historical_et:
        return current_ct
    ema = historical_et[-1]
    ema = alpha * current_ct + (1 - alpha) * ema
    return float(np.clip(ema, 0.0, 1.0))


def compute_v_personal(profile: UserProfile) -> float:
    """
    개인 취약성 V_personal ∈ [0, 1]
    건강보험공단 노인 질병 데이터 기반 고정 가중치.
    """
    score = 0.0

    # 연령 기반 (65세 이상부터 급증)
    if profile.age >= 75:
        score += 0.35
    elif profile.age >= 65:
        score += 0.25
    elif profile.age >= 50:
        score += 0.10

    # 기저질환
    if profile.has_hypertension:
        score += 0.20
    if profile.has_diabetes:
        score += 0.18
    if profile.has_heart_disease:
        score += 0.25
    if profile.has_respiratory:
        score += 0.15

    return float(np.clip(score, 0.0, 1.0))


def compute_social_vulnerability(region_code: str) -> float:
    """
    St: 사회 취약성 — 경로당 밀도, 독거노인 비율 집계.
    현재는 지역 코드별 고정 룩업 (실제 데이터 통합 시 교체).
    """
    # 서울 자치구별 독거노인 비율 근사 (2023 통계청)
    lookup = {
        "11110": 0.32, "11140": 0.28, "11170": 0.35, "11200": 0.30,
        "11215": 0.25, "11230": 0.27, "11260": 0.33, "11290": 0.29,
    }
    return lookup.get(region_code, 0.30)


def compute_resilience(region_code: str) -> float:
    """
    Rt: 회복력 — 의료기관 접근성 역수.
    높을수록 의료 접근성이 낮아 위험 (0~1).
    """
    # 단순 역수 근사: 의원 수 밀도 반영
    lookup = {
        "11110": 0.15, "11140": 0.20, "11170": 0.25, "11200": 0.18,
        "11215": 0.30, "11230": 0.22, "11260": 0.28, "11290": 0.19,
    }
    return lookup.get(region_code, 0.22)


@dataclass
class CHAIFeatureVector:
    ct: float
    bt: float
    et: float
    v_personal: float
    st: float
    rt: float

    def to_dict(self) -> dict:
        return {
            "ct": self.ct,
            "bt": self.bt,
            "et": self.et,
            "v_personal": self.v_personal,
            "st": self.st,
            "rt": self.rt,
        }

    def to_array(self) -> np.ndarray:
        return np.array([self.ct, self.bt, self.et, self.v_personal, self.st, self.rt])


def build_feature_vector(
    profile: UserProfile,
    climate: ClimateFeatures,
    bio: BioFeatures,
    historical_et: list[float] | None = None,
) -> CHAIFeatureVector:
    ct = compute_ct(climate)
    bt = compute_bt(bio)
    et = compute_et(historical_et or [], ct)
    v_personal = compute_v_personal(profile)
    st = compute_social_vulnerability(profile.region_code)
    rt = compute_resilience(profile.region_code)
    return CHAIFeatureVector(ct=ct, bt=bt, et=et, v_personal=v_personal, st=st, rt=rt)
