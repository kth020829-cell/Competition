from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# 배포 파라미터로 실수로 문자 그대로의 "mock" 이 들어오는 경우가 흔해서,
# 빈 문자열과 동일하게 목(mock) 모드 트리거로 인정한다.
_MOCK_SENTINELS = {"mock", "none", "dummy"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    env: str = "local"
    aws_access_key_id: str = "dummy"
    aws_secret_access_key: str = "dummy"
    aws_default_region: str = "ap-northeast-2"
    dynamodb_endpoint: str | None = None
    kma_api_key: str = ""
    airkorea_api_key: str = ""

    @field_validator("kma_api_key", "airkorea_api_key", mode="after")
    @classmethod
    def _treat_mock_sentinel_as_empty(cls, v: str) -> str:
        return "" if v.strip().lower() in _MOCK_SENTINELS else v

    # CHAI 가중치 (문헌 기반 고정값)
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2

    model_artifact_dir: str = "artifacts"

    @property
    def is_local(self) -> bool:
        return self.env == "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
