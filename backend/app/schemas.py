from typing import Literal

from pydantic import BaseModel, Field


class OrderItem(BaseModel):
    menu_id: str
    menu_name: str
    quantity: int = Field(ge=1, le=20)
    options: list[str] = Field(default_factory=list)


class MenuRecommendation(BaseModel):
    menu_id: str
    menu_name: str
    price: int = Field(ge=0)
    description: str
    reason: str
    caffeine_level: Literal["regular", "decaf", "none", "unknown"] = "unknown"
    caffeine_note: str = ""
    temperature_tags: list[str] = Field(default_factory=list)
    taste_tags: list[str] = Field(default_factory=list)
    allowed_options: list[str] = Field(default_factory=list)


class OrderInterpretation(BaseModel):
    intent: Literal[
        "order",
        "clarification",
        "menu_inquiry",
        "recommendation",
        "non_order",
    ]
    standard_order: str = ""
    items: list[OrderItem] = Field(default_factory=list)
    recommendations: list[MenuRecommendation] = Field(default_factory=list)
    suggested_tags: list[str] = Field(default_factory=list)
    unknown_terms: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    needs_clarification: bool
    clarification_question: str | None = None
    confidence: float = Field(ge=0, le=1)
    summary: str


ORDER_JSON_SCHEMA = OrderInterpretation.model_json_schema()
