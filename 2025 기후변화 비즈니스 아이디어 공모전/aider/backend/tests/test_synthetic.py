"""합성 데이터 생성기 + 피처 엔지니어링 단위 테스트."""
import pytest
import numpy as np
from app.services.synthetic_generator import WearableSimulator, IndoorEnvironmentSimulator, generate_full_scenario
from app.services.feature_engineering import (
    UserProfile, ClimateFeatures, BioFeatures,
    compute_ct, compute_bt, compute_et, compute_v_personal, build_feature_vector,
)


class TestWearableSimulator:
    def test_output_columns(self):
        sim = WearableSimulator()
        df = sim.generate("none", duration_min=10, seed=0)
        assert set(["hr_bpm", "hrv_ms", "skin_temp_c", "spo2_pct"]).issubset(df.columns)

    def test_heat_stress_raises_hr(self):
        sim = WearableSimulator()
        df_none = sim.generate("none", duration_min=60, seed=42)
        df_heat = sim.generate("heat", duration_min=60, seed=42)
        assert df_heat["hr_bpm"].mean() > df_none["hr_bpm"].mean()

    def test_heat_stress_lowers_hrv(self):
        sim = WearableSimulator()
        df_none = sim.generate("none", duration_min=60, seed=42)
        df_heat = sim.generate("heat", duration_min=60, seed=42)
        assert df_heat["hrv_ms"].mean() < df_none["hrv_ms"].mean()

    def test_spo2_clipped(self):
        sim = WearableSimulator()
        for st in ["none", "heat", "cold"]:
            df = sim.generate(st, duration_min=30, seed=1)
            assert (df["spo2_pct"] >= 90).all()
            assert (df["spo2_pct"] <= 100).all()


class TestIndoorEnvironmentSimulator:
    def test_rc_lag(self):
        """실내 온도가 외기 온도보다 지연되어야 함."""
        sim = IndoorEnvironmentSimulator(tau_hours=2.0)
        outdoor = [35.0] * 6 + [20.0] * 6
        df = sim.simulate(outdoor, sample_rate_sec=600)
        # 갑자기 냉각된 직후 실내는 여전히 높아야 함
        assert df.iloc[6]["indoor_temp_c"] > df.iloc[6]["outdoor_temp_c"]

    def test_indoor_humidity_clipped(self):
        sim = IndoorEnvironmentSimulator()
        outdoor = [25.0] * 12
        df = sim.simulate(outdoor)
        assert (df["indoor_rh_pct"] >= 20).all()
        assert (df["indoor_rh_pct"] <= 90).all()


class TestFeatureEngineering:
    def _make_climate(self, temp=25.0, cai=50.0, heat_warn=0):
        return ClimateFeatures(
            temp_c=temp, humidity_pct=60, wind_speed_ms=2, feels_like_c=temp,
            pm25=15, pm10=30, cai_index=cai, heat_warning_level=heat_warn,
        )

    def _make_bio(self, hr=72.0, hrv=55.0):
        return BioFeatures(hr_bpm=hr, hrv_ms=hrv, skin_temp_c=36.6, spo2_pct=98.0)

    def test_ct_range(self):
        for temp in [-20, 0, 25, 35, 42]:
            ct = compute_ct(self._make_climate(temp=temp))
            assert 0.0 <= ct <= 1.0

    def test_ct_heat_warning_increases_ct(self):
        ct_no_warn = compute_ct(self._make_climate(temp=34, heat_warn=0))
        ct_warn = compute_ct(self._make_climate(temp=34, heat_warn=1))
        assert ct_warn > ct_no_warn

    def test_bt_range(self):
        for hr in [40, 72, 120, 180]:
            bt = compute_bt(self._make_bio(hr=hr))
            assert 0.0 <= bt <= 1.0

    def test_bt_stressed_higher(self):
        bt_normal = compute_bt(self._make_bio(hr=72, hrv=55))
        bt_stressed = compute_bt(self._make_bio(hr=120, hrv=20))
        assert bt_stressed > bt_normal

    def test_et_ema(self):
        et = compute_et([0.3, 0.4, 0.5], current_ct=0.8, alpha=0.3)
        assert 0.0 <= et <= 1.0
        # EMA는 과거값(0.5)에 현재(0.8) 혼합 → 0.5와 0.8 사이
        assert 0.5 <= et <= 0.8

    def test_v_personal_elderly_diseased(self):
        elderly = UserProfile("x", age=80, gender="M", has_heart_disease=True)
        young = UserProfile("y", age=30, gender="F")
        assert compute_v_personal(elderly) > compute_v_personal(young)

    def test_build_feature_vector_all_in_range(self):
        profile = UserProfile("u1", age=70, gender="F", has_hypertension=True)
        climate = self._make_climate(temp=38, heat_warn=1)
        bio = self._make_bio(hr=110, hrv=25)
        fv = build_feature_vector(profile, climate, bio)
        arr = fv.to_array()
        assert (arr >= 0).all() and (arr <= 1).all()


class TestFullScenario:
    def test_generate_full_scenario_keys(self):
        result = generate_full_scenario("heat", seed=0)
        assert "wearable" in result and "indoor" in result

    def test_generate_full_scenario_nonempty(self):
        result = generate_full_scenario("cold", seed=1)
        assert len(result["wearable"]) > 0
        assert len(result["indoor"]) > 0
