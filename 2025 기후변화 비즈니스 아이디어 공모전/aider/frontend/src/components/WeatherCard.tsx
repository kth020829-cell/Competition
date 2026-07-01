import React from "react";

interface Props {
  temp: number | null;
  humidity: number | null;
  windSpeed: number | null;
  feelsLike: number | null;
  stressType: string;
}

const STRESS_EMOJI: Record<string, string> = {
  heat: "🌡️ 폭염",
  cold: "🌨️ 한파",
  none: "☁️ 보통",
};

export default function WeatherCard({ temp, humidity, windSpeed, feelsLike, stressType }: Props) {
  const fmt = (v: number | null, unit: string) =>
    v !== null ? `${v}${unit}` : "—";

  return (
    <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
      <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 16, color: "#e2e8f0" }}>
        기상 정보 {STRESS_EMOJI[stressType] ?? ""}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {[
          { label: "기온", value: fmt(temp, "°C"), color: "#f97316" },
          { label: "체감온도", value: fmt(feelsLike, "°C"), color: "#fb923c" },
          { label: "습도", value: fmt(humidity, "%"), color: "#38bdf8" },
          { label: "풍속", value: fmt(windSpeed, "m/s"), color: "#94a3b8" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ background: "#0f172a", borderRadius: 8, padding: "12px 14px" }}>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
