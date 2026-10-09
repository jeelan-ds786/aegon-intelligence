from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aegon_ingestion.documents import DocumentBlock, ParsedDocument


class JsonExportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    fields: dict[str, str]


class JsonExport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    documents: tuple[JsonExportRecord, ...]


class WeightedField(BaseModel):
    """An allowlisted JSON export field and its retrieval-time importance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    weight: float = Field(gt=0)


def parse_json_export(
    source_id: str,
    content: str | bytes,
    *,
    fields: tuple[WeightedField, ...],
) -> ParsedDocument:
    """Validate and parse a versioned document export with explicit field weights."""
    export = JsonExport.model_validate_json(content)
    blocks: list[DocumentBlock] = []
    for record_index, record in enumerate(export.documents):
        base = f"/documents/{record_index}"
        blocks.append(
            DocumentBlock(
                kind="heading",
                text=record.title,
                locator=f"{base}/title",
                heading_level=1,
            )
        )
        for field_config in fields:
            value = record.fields.get(field_config.name, "").strip()
            if value:
                blocks.append(
                    DocumentBlock(
                        kind="paragraph",
                        text=value,
                        locator=f"{base}/fields/{field_config.name}",
                        weight=field_config.weight,
                    )
                )
    return ParsedDocument(source_id=source_id, blocks=tuple(blocks), media_type="application/json")
