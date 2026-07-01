import React from "react";
import { PieChart, Pie, Cell, Tooltip } from "recharts";

interface Props {
  score: number;
  riskLevel: string;
  riskLabelKo: string;
}

const RISK_COLOR: Record<string, string> = {
  low: "#22c55e",
  moderate: "#f59e0b",
  high: "#f97316",
  critical: "#ef4444",
};

const RISK_BG: Record<string, string> = {
  low: "#14532d",
  moderate: "#451a03",
  high: "#431407",
  critical: "#450a0a",
};

export default function RiskGauge({ score, riskLevel, riskLabelKo }: Props) {
  const pct = Math.round(score * 100);
  const color = RISK_COLOR[riskLevel] ?? "#94a3b8";
  const bg = RISK_BG[riskLevel] ?? "#1e293b";

  const data = [
    { value: pct },
    { value: 100 - pct },
  ];

  return (
    <div style={{
      background: bg,
      borderRadius: 16,
      padding: "24px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      border: `2px solid ${color}`,
      minWidth: 220,
    }}>
      <div style={{ position: "relative" }}>
        <PieChart width={180} height={180}>
          <Pie
            data={data}
            cx={85}
            cy={85}
            innerRadius={60}
            outerRadius={80}
            startAngle={90}
            endAngle={-270}
            dataKey="value"
            strokeWidth={0}
          >
            <Cell fill={color} />
            <Cell fill="#1e293b" />
          </Pie>
        </PieChart>
        <div style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          textAlign: "center",
        }}>
          <div style={{ fontSize: 32, fontWeight: 700, color }}>{pct}</div>
          <div style={{ fontSize: 12, color: "#94a3b8" }}>/ 100</div>
        </div>
      </div>
      <div style={{ marginTop: 8, fontSize: 20, fontWeight: 700, color }}>
        {riskLabelKo}
      </div>
      <div style={{ fontSize: 13, color: "#94a3b8", marginTop: 4 }}>
        CHAI 위험지수
      </div>
    </div>
  );
}
