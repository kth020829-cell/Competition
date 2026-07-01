"""AWS Lambda 진입점 (Mangum 어댑터)."""
from mangum import Mangum
from app.main import app

handler = Mangum(app, lifespan="off")
