import React from "react";

interface Props {
  hr: number;
  hrv: number;
  skinTemp: number;
  spo2: number;
  isAnomaly: boolean;
}

export default function BioCard({ hr, hrv, skinTemp, spo2, isAnomaly }: Props) {
  return (
    <div style={{
      background: "#1e293b",
      borderRadius: 12,
      padding: 20,
      border: isAnomaly ? "1px solid #ef4444" : "none",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <span style={{ fontSize: 15, fontWeight: 600, color: "#e2e8f0" }}>생체신호 (시뮬레이션)</span>
        {isAnomaly && (
          <span style={{
            background: "#450a0a",
            color: "#ef4444",
            fontSize: 11,
            padding: "2px 8px",
            borderRadius: 4,
            fontWeight: 600,
          }}>
            이상 탐지
          </span>
        )}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {[
          { label: "심박수", value: `${hr.toFixed(0)} bpm`, color: "#f43f5e", normal: hr >= 60 && hr <= 100 },
          { label: "HRV (RMSSD)", value: `${hrv.toFixed(0)} ms`, color: "#a78bfa", normal: hrv >= 40 },
          { label: "피부 체온", value: `${skinTemp.toFixed(1)} °C`, color: "#f97316", normal: skinTemp >= 36.1 && skinTemp <= 37.2 },
          { label: "산소포화도", value: `${spo2.toFixed(1)} %`, color: "#22d3ee", normal: spo2 >= 95 },
        ].map(({ label, value, color, normal }) => (
          <div key={label} style={{ background: "#0f172a", borderRadius: 8, padding: "12px 14px" }}>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 20, fontWeight: 700, color }}>{value}</div>
            <div style={{ fontSize: 10, color: normal ? "#22c55e" : "#ef4444", marginTop: 2 }}>
              {normal ? "정상 범위" : "범위 이탈"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
