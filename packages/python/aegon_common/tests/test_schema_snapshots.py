import json
import os
from pathlib import Path

import pytest
from aegon_common import (
    AttachmentRef,
    ChatRequest,
    ChatResponse,
    Chunk,
    Citation,
    ErrorCode,
    Evidence,
    Modality,
    StreamEvent,
)
from pydantic import TypeAdapter
from pydantic.json_schema import JsonSchemaValue

CONTRACTS_DIR = Path(__file__).parents[4] / "docs" / "contracts"
SCHEMAS: dict[str, JsonSchemaValue] = {
    "attachment-ref": AttachmentRef.model_json_schema(),
    "chat-request": ChatRequest.model_json_schema(),
    "chat-response": ChatResponse.model_json_schema(),
    "chunk": Chunk.model_json_schema(),
    "citation": Citation.model_json_schema(),
    "error-code": TypeAdapter(ErrorCode).json_schema(),
    "evidence": Evidence.model_json_schema(),
    "modality": TypeAdapter(Modality).json_schema(),
    "stream-event": TypeAdapter(StreamEvent).json_schema(),
}


@pytest.mark.parametrize(("name", "schema"), SCHEMAS.items())
def test_json_schema_matches_snapshot(name: str, schema: JsonSchemaValue) -> None:
    snapshot_path = CONTRACTS_DIR / f"{name}.schema.json"
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"

    if os.environ.get("UPDATE_CONTRACT_SCHEMAS") == "1":
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(rendered, encoding="utf-8")

    assert snapshot_path.read_text(encoding="utf-8") == rendered
