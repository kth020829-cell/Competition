import React, { useState, useEffect } from "react";
import RiskGauge from "./components/RiskGauge";
import FeatureBar from "./components/FeatureBar";
import WeatherCard from "./components/WeatherCard";
import BioCard from "./components/BioCard";
import AlertBanner from "./components/AlertBanner";
import { useSimulation } from "./hooks/useSimulation";

type StressType = "none" | "heat" | "cold";

const STRESS_OPTIONS: { value: StressType; label: string; temp: number }[] = [
  { value: "heat", label: "폭염 시나리오 (35°C)", temp: 35 },
  { value: "cold", label: "한파 시나리오 (-5°C)", temp: -5 },
  { value: "none", label: "정상 날씨 (22°C)", temp: 22 },
];

export default function App() {
  const [stressType, setStressType] = useState<StressType>("heat");
  const [duration, setDuration] = useState(6);
  const { result, forecast, warnings, loading, error, run } = useSimulation();

  useEffect(() => {
    const opt = STRESS_OPTIONS.find(o => o.value === stressType)!;
    run(stressType, opt.temp, duration);
  }, []);  // eslint-disable-line

  const handleRun = () => {
    const opt = STRESS_OPTIONS.find(o => o.value === stressType)!;
    run(stressType, opt.temp, duration);
  };

  return (
    <div style={{ minHeight: "100vh", padding: "24px 20px", maxWidth: 1100, margin: "0 auto" }}>
      {/* 헤더 */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 26, fontWeight: 800, color: "#f1f5f9" }}>
          Aider
          <span style={{ fontSize: 14, fontWeight: 400, color: "#64748b", marginLeft: 12 }}>
            기후적응 헬스케어 플랫폼 — CHAI 모델 데모
          </span>
        </h1>
        <div style={{ fontSize: 12, color: "#475569", marginTop: 4 }}>
          시뮬레이션 모드 | 실제 사용자 데이터 없이 동작 | 국가기후위기적응센터 AI
        </div>
      </div>

      {/* 컨트롤 패널 */}
      <div style={{
        background: "#1e293b", borderRadius: 12, padding: 20, marginBottom: 24,
        display: "flex", flexWrap: "wrap", gap: 16, alignItems: "flex-end",
      }}>
        <div>
          <label style={{ fontSize: 12, color: "#94a3b8", display: "block", marginBottom: 6 }}>시나리오 선택</label>
          <select
            value={stressType}
            onChange={e => setStressType(e.target.value as StressType)}
            style={{
              background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155",
              borderRadius: 6, padding: "8px 12px", fontSize: 14, cursor: "pointer",
            }}
          >
            {STRESS_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label style={{ fontSize: 12, color: "#94a3b8", display: "block", marginBottom: 6 }}>
            시뮬레이션 시간: {duration}시간
          </label>
          <input
            type="range" min={1} max={12} value={duration}
            onChange={e => setDuration(Number(e.target.value))}
            style={{ width: 160 }}
          />
        </div>
        <button
          onClick={handleRun}
          disabled={loading}
          style={{
            background: loading ? "#334155" : "#3b82f6",
            color: "#fff", border: "none", borderRadius: 8,
            padding: "9px 20px", fontSize: 14, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "계산 중..." : "CHAI 계산"}
        </button>
      </div>

      {error && (
        <div style={{ background: "#450a0a", border: "1px solid #ef4444", borderRadius: 8, padding: 14, marginBottom: 20, color: "#fca5a5", fontSize: 13 }}>
          오류: {error} — 백엔드가 실행 중인지 확인하세요 (http://localhost:8000)
        </div>
      )}

      {result && (
        <>
          {/* 경보 배너 */}
          <div style={{ marginBottom: 20 }}>
            <AlertBanner
              warnings={warnings?.warnings ?? []}
              isAnomaly={result.anomaly_detection.is_anomaly}
              riskLevel={result.risk_level}
            />
          </div>

          {/* 메인 그리드 */}
          <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 20, marginBottom: 20 }}>
            <RiskGauge
              score={result.score}
              riskLevel={result.risk_level}
              riskLabelKo={result.risk_label_ko}
            />
            <FeatureBar features={result.features} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            <WeatherCard
              temp={forecast?.forecast.temp_c ?? null}
              humidity={forecast?.forecast.humidity_pct ?? null}
              windSpeed={forecast?.forecast.wind_speed_ms ?? null}
              feelsLike={forecast?.forecast.feels_like_c ?? null}
              stressType={result.stress_type}
            />
            <BioCard
              hr={result.simulated_bio.hr_bpm}
              hrv={result.simulated_bio.hrv_ms}
              skinTemp={result.simulated_bio.skin_temp_c}
              spo2={result.simulated_bio.spo2_pct}
              isAnomaly={result.anomaly_detection.is_anomaly}
            />
          </div>

          {/* 이상탐지 상세 */}
          <div style={{ background: "#1e293b", borderRadius: 12, padding: 16, marginTop: 20 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 10 }}>이상탐지 모듈 (오토인코더)</div>
            <div style={{ display: "flex", gap: 24, fontSize: 12, color: "#94a3b8" }}>
              <span>모델 로드: <b style={{ color: result.anomaly_detection.model_loaded ? "#22c55e" : "#f59e0b" }}>
                {result.anomaly_detection.model_loaded ? "완료" : "미학습 (학습 스크립트 실행 필요)"}
              </b></span>
              <span>재구성 오차: <b style={{ color: "#e2e8f0" }}>
                {result.anomaly_detection.recon_error?.toFixed(4) ?? "N/A"}
              </b></span>
              <span>임계값: <b style={{ color: "#e2e8f0" }}>{result.anomaly_detection.threshold.toFixed(4)}</b></span>
            </div>
          </div>

          <div style={{ marginTop: 14, fontSize: 11, color: "#334155", textAlign: "right" }}>
            계산 시각: {new Date(result.timestamp).toLocaleString("ko-KR")} | 데이터 소스: 시뮬레이션
          </div>
        </>
      )}

      {!result && !loading && (
        <div style={{ textAlign: "center", padding: "60px 0", color: "#475569" }}>
          위에서 시나리오를 선택하고 CHAI 계산 버튼을 누르세요
        </div>
      )}
    </div>
  );
}
