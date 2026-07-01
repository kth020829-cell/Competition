import React from "react";

interface FeatureRow {
  label: string;
  value: number;
  color: string;
  description: string;
}

interface Props {
  features: {
    ct: number; bt: number; et: number;
    v_personal: number; st: number; rt: number;
  };
}

const FEATURE_META: FeatureRow[] = [
  { label: "Ct — 기후 스트레스", value: 0, color: "#f97316", description: "현재 기상 조건의 위험도" },
  { label: "Bt — 생체 반응", value: 0, color: "#a78bfa", description: "생체신호 이탈 정도" },
  { label: "Et — 환경 노출", value: 0, color: "#38bdf8", description: "누적 노출 이력 (EMA)" },
  { label: "V_personal — 개인 취약성", value: 0, color: "#fb7185", description: "연령·기저질환 기반" },
  { label: "St — 사회 취약성", value: 0, color: "#facc15", description: "지역 독거노인 비율 등" },
  { label: "Rt — 회복력", value: 0, color: "#4ade80", description: "의료 접근성 (역수)" },
];

export default function FeatureBar({ features }: Props) {
  const values = [features.ct, features.bt, features.et, features.v_personal, features.st, features.rt];

  return (
    <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
      <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 16, color: "#e2e8f0" }}>
        CHAI 피처 분해
      </div>
      {FEATURE_META.map((meta, idx) => {
        const pct = Math.round(values[idx] * 100);
        return (
          <div key={meta.label} style={{ marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ fontSize: 12, color: "#cbd5e1" }}>{meta.label}</span>
              <span style={{ fontSize: 12, fontWeight: 600, color: meta.color }}>{pct}%</span>
            </div>
            <div style={{ background: "#0f172a", borderRadius: 4, height: 8, overflow: "hidden" }}>
              <div style={{
                width: `${pct}%`,
                height: "100%",
                background: meta.color,
                borderRadius: 4,
                transition: "width 0.6s ease",
              }} />
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{meta.description}</div>
          </div>
        );
      })}
    </div>
  );
}
