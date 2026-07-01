from fastapi import APIRouter, Query
from app.services.data_collector import KMAClient, parse_forecast

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/warnings")
async def get_weather_warnings(region_code: str = Query("11")):
    """기상청 폭염/한파 특보 조회."""
    kma = KMAClient()
    raw = await kma.get_special_warning(region_code=region_code)
    items = raw.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    warnings = []
    for item in items if isinstance(items, list) else []:
        warnings.append({
            "type": item.get("wrnId"),
            "level": item.get("lvl"),
            "message": item.get("wrnMsg"),
        })
    return {
        "region_code": region_code,
        "warnings": warnings,
        "source": "mock" if raw.get("_mock") else "kma",
    }


@router.get("/forecast")
async def get_forecast(
    nx: int = Query(60),
    ny: int = Query(127),
):
    """동네예보 조회."""
    kma = KMAClient()
    raw = await kma.get_vilage_forecast(nx=nx, ny=ny)
    fc = parse_forecast(raw)
    return {
        "forecast": fc,
        "source": "mock" if raw.get("_mock") else "kma",
    }
