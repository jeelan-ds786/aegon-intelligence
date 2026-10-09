from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BlockKind = Literal["heading", "paragraph", "code", "table"]


class DocumentBlock(BaseModel):
    """A source-aligned unit that a chunker may keep intact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BlockKind
    text: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    heading_level: int | None = Field(default=None, ge=1, le=6)
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_heading_level(self) -> "DocumentBlock":
        if (self.kind == "heading") != (self.heading_level is not None):
            raise ValueError("heading_level is required only for heading blocks")
        return self


class ParsedDocument(BaseModel):
    """Deterministic parser output before chunking or redaction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    blocks: tuple[DocumentBlock, ...]
    media_type: str = Field(min_length=1)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
