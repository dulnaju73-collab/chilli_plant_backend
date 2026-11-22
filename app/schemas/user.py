from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserProfile(BaseModel):
    uid: str
    username: str
    email: str | None = None
    created_at: datetime | None = None


class UserProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=40, pattern=r"^[\w .'-]+$")
