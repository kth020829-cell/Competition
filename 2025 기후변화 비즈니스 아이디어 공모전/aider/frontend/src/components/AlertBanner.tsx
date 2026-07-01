import React from "react";

interface Warning {
  type: string;
  level: string;
  message: string;
}

interface Props {
  warnings: Warning[];
  isAnomaly: boolean;
  riskLevel: string;
}

const LEVEL_MAP: Record<string, string> = { "1": "주의보", "2": "경보" };
const WARN_TYPE_MAP: Record<string, string> = {
  "W01": "폭염", "W02": "폭염", "W03": "한파", "W04": "한파",
};

export default function AlertBanner({ warnings, isAnomaly, riskLevel }: Props) {
  const showCritical = riskLevel === "critical" || riskLevel === "high";

  if (warnings.length === 0 && !isAnomaly && !showCritical) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {showCritical && (
        <div style={{
          background: riskLevel === "critical" ? "#450a0a" : "#431407",
          border: `1px solid ${riskLevel === "critical" ? "#ef4444" : "#f97316"}`,
          borderRadius: 8,
          padding: "10px 16px",
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}>
          <span style={{ fontSize: 18 }}>{riskLevel === "critical" ? "🚨" : "⚠️"}</span>
          <div>
            <div style={{ fontWeight: 600, color: riskLevel === "critical" ? "#ef4444" : "#f97316", fontSize: 13 }}>
              {riskLevel === "critical" ? "CHAI 위험 수준: 위험" : "CHAI 위험 수준: 높음"}
            </div>
            <div style={{ color: "#cbd5e1", fontSize: 12 }}>즉각적인 조치가 필요합니다. 경로당 또는 의료기관 방문을 권장합니다.</div>
          </div>
        </div>
      )}
      {isAnomaly && (
        <div style={{
          background: "#1c1917",
          border: "1px solid #a8a29e",
          borderRadius: 8,
          padding: "10px 16px",
          display: "flex",
          gap: 10,
          alignItems: "center",
        }}>
          <span style={{ fontSize: 18 }}>🔍</span>
          <div>
            <div style={{ fontWeight: 600, color: "#d6d3d1", fontSize: 13 }}>오토인코더 이상 탐지</div>
            <div style={{ color: "#a8a29e", fontSize: 12 }}>생체신호 패턴에서 급변이 감지되었습니다.</div>
          </div>
        </div>
      )}
      {warnings.map((w, i) => (
        <div key={i} style={{
          background: "#1c0a00",
          border: "1px solid #f59e0b",
          borderRadius: 8,
          padding: "10px 16px",
          display: "flex",
          gap: 10,
          alignItems: "center",
        }}>
          <span style={{ fontSize: 18 }}>🌤️</span>
          <div>
            <div style={{ fontWeight: 600, color: "#fcd34d", fontSize: 13 }}>
              기상청 {WARN_TYPE_MAP[w.type] ?? ""} {LEVEL_MAP[w.level] ?? w.level}
            </div>
            <div style={{ color: "#fde68a", fontSize: 12 }}>{w.message}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
