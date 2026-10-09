import pytest
from aegon_ingestion import DocumentBlock, ParsedDocument
from aegon_ingestion.chunker import Chunker, OversizedAtomicBlockError


def _document(*blocks: DocumentBlock) -> ParsedDocument:
    return ParsedDocument(source_id="source-1", blocks=blocks, media_type="text/markdown")


def test_heading_hierarchy_parent_ids_and_ids_are_stable() -> None:
    document = _document(
        DocumentBlock(kind="heading", text="Guide", locator="line:1", heading_level=1),
        DocumentBlock(kind="paragraph", text="intro text", locator="line:2"),
        DocumentBlock(kind="heading", text="Install", locator="line:4", heading_level=2),
        DocumentBlock(kind="paragraph", text="run command", locator="line:5"),
    )
    chunker = Chunker()

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert first == second
    assert [chunk.heading_path for chunk in first] == [("Guide",), ("Guide", "Install")]
    assert first[0].parent_id != first[1].parent_id


def test_long_prose_uses_bounded_intra_section_overlap() -> None:
    words = " ".join(f"word-{index}" for index in range(900))
    document = _document(
        DocumentBlock(kind="heading", text="Long", locator="line:1", heading_level=1),
        DocumentBlock(kind="paragraph", text=words, locator="line:2"),
    )

    chunks = Chunker().chunk(document)

    assert len(chunks) == 3
    assert all(chunk.token_count <= 800 for chunk in chunks)
    assert chunks[1].overlap_tokens == 48
    assert chunks[2].overlap_tokens == 48


def test_code_blocks_stay_intact_and_fail_when_over_hard_max() -> None:
    code = "```\n" + " ".join("x" for _ in range(500)) + "\n```"
    chunks = Chunker().chunk(_document(DocumentBlock(kind="code", text=code, locator="lines:1-3")))
    assert chunks[0].text == code

    oversized = " ".join("x" for _ in range(801))
    with pytest.raises(OversizedAtomicBlockError, match="maximum is 800"):
        Chunker().chunk(_document(DocumentBlock(kind="table", text=oversized, locator="lines:1-3")))
