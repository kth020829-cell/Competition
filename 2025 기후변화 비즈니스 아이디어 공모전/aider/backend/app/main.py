from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import get_settings
from app.db.dynamodb import ensure_tables
from app.models import lgbm_model, autoencoder, transformer_encoder
from app.api.routes import health, users, predictions, alerts


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # 로컬: DynamoDB 테이블 자동 생성
    if settings.is_local:
        try:
            ensure_tables()
        except Exception as e:
            print(f"[WARN] DynamoDB setup failed (DynamoDB-local not running?): {e}")

    # 모델 사전 로드
    lgbm_model.load_model()
    autoencoder.load_autoencoder()
    transformer_encoder.load_transformer()

    yield


app = FastAPI(
    title="Aider — 기후적응 헬스케어 플랫폼",
    description="CHAI 모델 기반 기후건강 위험지수 예측 API (시뮬레이션 모드 포함)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(users.router)
app.include_router(predictions.router)
app.include_router(alerts.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Starlette 의 기본 미처리-예외 응답 경로는 CORSMiddleware 를 거치지 않아
    # 브라우저에서 CORS 오류(Network Error)로 오인되므로, 여기서 명시적으로 처리한다.
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
def root():
    return {
        "service": "Aider API",
        "version": "1.0.0",
        "docs": "/docs",
        "note": "시뮬레이션 모드 — 실제 사용자 데이터 없이 동작",
    }
