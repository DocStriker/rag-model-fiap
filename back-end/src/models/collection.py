from pydantic import BaseModel, Field


class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_\-]+$")
    vector_size: int = Field(default=1536, ge=1, le=65536)


class CollectionListResponse(BaseModel):
    collections: list[str]


class UploadResponse(BaseModel):
    collection: str
    chunks: int
