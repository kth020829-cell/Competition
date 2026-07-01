import axios from "axios";

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL, timeout: 15000 });

export interface SimulateResult {
  score: number;
  risk_level: "low" | "moderate" | "high" | "critical";
  risk_label_ko: string;
  features: {
    ct: number; bt: number; et: number;
    v_personal: number; st: number; rt: number;
  };
  anomaly_detection: {
    is_anomaly: boolean;
    recon_error: number | null;
    threshold: number;
    model_loaded: boolean;
  };
  simulated_bio: {
    hr_bpm: number; hrv_ms: number; skin_temp_c: number; spo2_pct: number;
  };
  stress_type: string;
  simulation_mode: boolean;
  timestamp: string;
}

export interface ForecastResult {
  forecast: {
    temp_c: number | null;
    humidity_pct: number | null;
    wind_speed_ms: number | null;
    feels_like_c: number | null;
  };
  source: string;
}

export interface WarningResult {
  region_code: string;
  warnings: { type: string; level: string; message: string }[];
  source: string;
}

export const simulateScenario = async (
  stressType: "none" | "heat" | "cold",
  outdoorTemp: number,
  durationHours: number
): Promise<SimulateResult> => {
  const { data } = await api.get("/predictions/simulate", {
    params: { stress_type: stressType, outdoor_temp: outdoorTemp, duration_hours: durationHours },
  });
  return data;
};

export const getForecast = async (nx = 60, ny = 127): Promise<ForecastResult> => {
  const { data } = await api.get("/alerts/forecast", { params: { nx, ny } });
  return data;
};

export const getWarnings = async (regionCode = "11"): Promise<WarningResult> => {
  const { data } = await api.get("/alerts/warnings", { params: { region_code: regionCode } });
  return data;
};

export const getHealth = async () => {
  const { data } = await api.get("/health");
  return data;
};
