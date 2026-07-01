import { useState, useCallback } from "react";
import { simulateScenario, getForecast, getWarnings, SimulateResult, ForecastResult, WarningResult } from "../services/api";

type StressType = "none" | "heat" | "cold";

interface SimulationState {
  result: SimulateResult | null;
  forecast: ForecastResult | null;
  warnings: WarningResult | null;
  loading: boolean;
  error: string | null;
}

export function useSimulation() {
  const [state, setState] = useState<SimulationState>({
    result: null,
    forecast: null,
    warnings: null,
    loading: false,
    error: null,
  });

  const run = useCallback(async (
    stressType: StressType,
    outdoorTemp: number,
    durationHours: number
  ) => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const [result, forecast, warnings] = await Promise.all([
        simulateScenario(stressType, outdoorTemp, durationHours),
        getForecast(),
        getWarnings(),
      ]);
      setState({ result, forecast, warnings, loading: false, error: null });
    } catch (e: any) {
      setState(s => ({
        ...s,
        loading: false,
        error: e?.response?.data?.detail ?? e.message ?? "API 오류",
      }));
    }
  }, []);

  return { ...state, run };
}
