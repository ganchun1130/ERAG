import json
from typing import Annotated

from pydantic import BaseModel, Field, JsonValue, StrictBool, StringConstraints, field_validator

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class Chunk(BaseModel):
    id: str
    content: str = Field(min_length=1)
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    source: str
    chapter: str = ""
    page: int | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "content", "source")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("知识块 id、content、source 不能只包含空白")
        return value

    @field_validator("metadata")
    @classmethod
    def finite_metadata(cls, value):
        json.dumps(value, allow_nan=False)
        return value


class SubQuery(BaseModel):
    query: QueryText
    chapter: str = ""


class QueryPlan(BaseModel):
    needs_retrieval: StrictBool = True
    question_type: str = "概念解释"
    key_query: QueryText
    sub_queries: list[SubQuery] = Field(min_length=1, max_length=5)


class Review(BaseModel):
    complete: StrictBool
    missing_queries: list[QueryText] = Field(default_factory=list, max_length=5)


class Enrichment(BaseModel):
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    tags: list[QueryText] = Field(min_length=1, max_length=3)


class GeneratedAnswer(BaseModel):
    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Hit(BaseModel):
    chunk: Chunk
    score: float = Field(allow_inf_nan=False)
    sub_query: str
    route: str


class Answer(BaseModel):
    answer: str
    sources: list[Hit] = Field(default_factory=list)
    plan: QueryPlan
    trace: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
