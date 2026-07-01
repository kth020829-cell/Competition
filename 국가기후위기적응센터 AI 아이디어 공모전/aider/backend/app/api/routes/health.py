from fastapi import APIRouter
from datetime import datetime, timezone
from app.models import lgbm_model, autoencoder, transformer_encoder

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            "lgbm": lgbm_model.get_model() is not None,
            "transformer": transformer_encoder.get_transformer() is not None,
            "autoencoder": autoencoder.get_autoencoder() is not None,
        },
    }
