import boto3
from boto3.dynamodb.conditions import Key
from app.config import get_settings
from typing import Any

_dynamodb = None


def get_dynamodb():
    global _dynamodb
    if _dynamodb is None:
        settings = get_settings()
        kwargs: dict[str, Any] = {
            "region_name": settings.aws_default_region,
            "aws_access_key_id": settings.aws_access_key_id,
            "aws_secret_access_key": settings.aws_secret_access_key,
        }
        if settings.dynamodb_endpoint:
            kwargs["endpoint_url"] = settings.dynamodb_endpoint
        _dynamodb = boto3.resource("dynamodb", **kwargs)
    return _dynamodb


def ensure_tables():
    """로컬 개발 시 테이블이 없으면 자동 생성."""
    db = get_dynamodb()

    existing = {t.name for t in db.tables.all()}

    tables = [
        {
            "TableName": "aider-users",
            "KeySchema": [{"AttributeName": "user_id", "KeyType": "HASH"}],
            "AttributeDefinitions": [{"AttributeName": "user_id", "AttributeType": "S"}],
            "BillingMode": "PAY_PER_REQUEST",
        },
        {
            "TableName": "aider-records",
            "KeySchema": [
                {"AttributeName": "user_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
            "BillingMode": "PAY_PER_REQUEST",
        },
        {
            "TableName": "aider-alerts",
            "KeySchema": [
                {"AttributeName": "region_code", "KeyType": "HASH"},
                {"AttributeName": "issued_at", "KeyType": "RANGE"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "region_code", "AttributeType": "S"},
                {"AttributeName": "issued_at", "AttributeType": "S"},
            ],
            "BillingMode": "PAY_PER_REQUEST",
        },
    ]

    for spec in tables:
        if spec["TableName"] not in existing:
            table = db.create_table(**spec)
            table.wait_until_exists()


def put_item(table_name: str, item: dict):
    db = get_dynamodb()
    table = db.Table(table_name)
    table.put_item(Item=item)


def get_item(table_name: str, key: dict) -> dict | None:
    db = get_dynamodb()
    table = db.Table(table_name)
    resp = table.get_item(Key=key)
    return resp.get("Item")


def query_items(table_name: str, key_condition, **kwargs) -> list[dict]:
    db = get_dynamodb()
    table = db.Table(table_name)
    resp = table.query(KeyConditionExpression=key_condition, **kwargs)
    return resp.get("Items", [])
