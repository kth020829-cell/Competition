"""FastAPI 엔드포인트 통합 테스트 (DynamoDB-local 없이 동작)."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # DynamoDB 호출을 목(mock)으로 대체
    with patch("app.db.dynamodb.ensure_tables"), \
         patch("app.db.dynamodb.get_dynamodb"):
        from app.main import app
        return TestClient(app)


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Aider" in resp.json()["service"]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_simulate_heat(client):
    resp = client.get("/predictions/simulate?stress_type=heat&outdoor_temp=37&duration_hours=2")
    assert resp.status_code == 200
    data = resp.json()
    assert "score" in data
    assert "risk_level" in data
    assert 0.0 <= data["score"] <= 1.0
    assert data["simulation_mode"] is True


def test_simulate_cold(client):
    resp = client.get("/predictions/simulate?stress_type=cold&outdoor_temp=-5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_level"] in ["low", "moderate", "high", "critical"]


def test_forecast_alert(client):
    resp = client.get("/alerts/forecast?nx=60&ny=127")
    assert resp.status_code == 200
    assert "forecast" in resp.json()


def test_warnings_alert(client):
    resp = client.get("/alerts/warnings?region_code=11")
    assert resp.status_code == 200
    assert "warnings" in resp.json()
