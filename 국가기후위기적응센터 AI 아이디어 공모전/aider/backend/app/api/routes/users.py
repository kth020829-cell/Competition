from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.db import dynamodb

router = APIRouter(prefix="/users", tags=["users"])


class UserCreateRequest(BaseModel):
    user_id: str
    name: str
    age: int = Field(ge=0, le=120)
    gender: str = Field(pattern="^[MF]$")
    region_code: str = "11"
    has_hypertension: bool = False
    has_diabetes: bool = False
    has_heart_disease: bool = False
    has_respiratory: bool = False


@router.post("", status_code=201)
def create_user(req: UserCreateRequest):
    existing = dynamodb.get_item("aider-users", {"user_id": req.user_id})
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")
    dynamodb.put_item("aider-users", req.model_dump())
    return {"message": "created", "user_id": req.user_id}


@router.get("/{user_id}")
def get_user(user_id: str):
    item = dynamodb.get_item("aider-users", {"user_id": user_id})
    if not item:
        raise HTTPException(status_code=404, detail="User not found")
    return item
