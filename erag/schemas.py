from typing import Literal
from pydantic import BaseModel, Field, StrictBool


class Chunk(BaseModel):
    id: str
    content: str = Field(min_length=1)
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    source: str
    chapter: str = ""
    page: int | None = None
    kind: Literal["local", "qa"] = "local"


class SubQuery(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    chapter: str = ""


class QueryPlan(BaseModel):
    needs_retrieval: StrictBool = True
    question_type: str = "概念解释"
    key_query: str = Field(min_length=1, max_length=1000)
    sub_queries: list[SubQuery] = Field(min_length=1, max_length=5)


class Review(BaseModel):
    complete: StrictBool
    missing_queries: list[str] = Field(default_factory=list, max_length=5)


class Hit(BaseModel):
    chunk: Chunk
    score: float
    sub_query: str
    route: str


class Answer(BaseModel):
    answer: str
    sources: list[Hit] = Field(default_factory=list)
    plan: QueryPlan
    trace: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
