"""
외부 API 수집기
- 기상청 동네예보 + 특보 API (공공데이터포털)
- 에어코리아 CAI (통합대기환경지수) API
"""
import httpx
import asyncio
from datetime import datetime
import pytz
from app.config import get_settings

KST = pytz.timezone("Asia/Seoul")


class KMAClient:
    """기상청 Open API 클라이언트."""

    BASE = "http://apis.data.go.kr/1360000"

    def __init__(self):
        self.key = get_settings().kma_api_key

    async def get_vilage_forecast(
        self,
        nx: int,
        ny: int,
        base_date: str | None = None,
        base_time: str = "0500",
    ) -> dict:
        """동네예보 단기 조회 (VilageFcstInfoService2.0)."""
        if base_date is None:
            base_date = datetime.now(KST).strftime("%Y%m%d")

        url = f"{self.BASE}/VilageFcstInfoService2.0/getVilageFcst"
        params = {
            "serviceKey": self.key,
            "numOfRows": 300,
            "pageNo": 1,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
        }

        if not self.key:
            return self._mock_forecast(nx, ny)

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    async def get_special_warning(self, region_code: str = "11") -> dict:
        """폭염/한파 특보 조회 (WthrWrnInfoService)."""
        if not self.key:
            return self._mock_warning(region_code)

        url = f"{self.BASE}/WthrWrnInfoService/getWthrWrnList"
        params = {
            "serviceKey": self.key,
            "numOfRows": 10,
            "pageNo": 1,
            "dataType": "JSON",
            "stnId": region_code,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    def _mock_forecast(self, nx: int, ny: int) -> dict:
        """API 키 없을 때 로컬 테스트용 목 데이터."""
        now = datetime.now(KST)
        return {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {"category": "TMP", "fcstValue": "33", "fcstDate": now.strftime("%Y%m%d"), "fcstTime": "1500"},
                            {"category": "REH", "fcstValue": "65", "fcstDate": now.strftime("%Y%m%d"), "fcstTime": "1500"},
                            {"category": "WSD", "fcstValue": "2.5", "fcstDate": now.strftime("%Y%m%d"), "fcstTime": "1500"},
                        ]
                    }
                }
            },
            "_mock": True,
        }

    def _mock_warning(self, region_code: str) -> dict:
        return {
            "response": {"body": {"items": {"item": [{"wrnId": "W02", "lvl": "1", "wrnMsg": "폭염주의보"}]}}},
            "_mock": True,
        }


class AirKoreaClient:
    """에어코리아 대기 정보 API 클라이언트."""

    BASE = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc"

    def __init__(self):
        self.key = get_settings().airkorea_api_key

    async def get_cai(self, station_name: str = "중구") -> dict:
        """측정소 기준 통합대기환경지수(CAI) 조회."""
        if not self.key:
            return self._mock_cai(station_name)

        url = f"{self.BASE}/getMsrstnAcctoRltmMesureDnsty"
        params = {
            "serviceKey": self.key,
            "returnType": "json",
            "numOfRows": 1,
            "pageNo": 1,
            "stationName": station_name,
            "dataTerm": "DAILY",
            "ver": "1.3",
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    def _mock_cai(self, station_name: str) -> dict:
        return {
            "response": {
                "body": {
                    "items": [
                        {
                            "stationName": station_name,
                            "pm10Value": "35",
                            "pm25Value": "18",
                            "o3Value": "0.045",
                            "no2Value": "0.025",
                            "coValue": "0.5",
                            "so2Value": "0.004",
                            "khaiValue": "75",
                            "khaiGrade": "2",
                        }
                    ]
                }
            },
            "_mock": True,
        }


def parse_forecast(raw: dict) -> dict:
    """동네예보 응답에서 핵심 기상 피처 추출."""
    items = raw.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    result: dict[str, float | str | None] = {
        "temp_c": None,
        "humidity_pct": None,
        "wind_speed_ms": None,
        "feels_like_c": None,
    }
    for item in items:
        cat = item.get("category")
        val = item.get("fcstValue")
        try:
            if cat == "TMP":
                result["temp_c"] = float(val)
            elif cat == "REH":
                result["humidity_pct"] = float(val)
            elif cat == "WSD":
                result["wind_speed_ms"] = float(val)
        except (TypeError, ValueError):
            pass

    # 체감온도 (Steadman 공식 근사)
    t = result["temp_c"]
    rh = result["humidity_pct"]
    ws = result["wind_speed_ms"]
    if t is not None and ws is not None:
        # 풍속 보정 (단순 선형)
        result["feels_like_c"] = round(t - 0.5 * ws, 1)
    return result


def parse_cai(raw: dict) -> dict:
    """에어코리아 응답에서 CAI 피처 추출."""
    items = raw.get("response", {}).get("body", {}).get("items", [])
    if not items:
        return {"pm10": None, "pm25": None, "o3": None, "cai_index": None, "cai_grade": None}
    it = items[0]

    def safe_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "pm10": safe_float(it.get("pm10Value")),
        "pm25": safe_float(it.get("pm25Value")),
        "o3": safe_float(it.get("o3Value")),
        "cai_index": safe_float(it.get("khaiValue")),
        "cai_grade": safe_float(it.get("khaiGrade")),
    }
