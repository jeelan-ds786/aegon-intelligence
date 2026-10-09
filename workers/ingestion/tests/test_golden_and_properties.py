import json
from pathlib import Path

from aegon_ingestion import Chunker, DocumentBlock, ParsedDocument, WhitespaceTokenizer
from aegon_ingestion.parsers import WeightedField, parse_html, parse_json_export, parse_markdown
from hypothesis import given
from hypothesis import strategies as st

FIXTURES = Path(__file__).parent / "fixtures"


def _snapshot(document: ParsedDocument) -> list[list[str | int | float | None]]:
    return [
        [block.kind, block.text, block.locator, block.heading_level, block.weight]
        for block in document.blocks
    ]


def test_five_diverse_documents_match_parser_goldens() -> None:
    parsed = {
        "guide.md": parse_markdown("guide", (FIXTURES / "guide.md").read_text()),
        "component.mdx": parse_markdown(
            "component", (FIXTURES / "component.mdx").read_text(), mdx=True
        ),
        "article.html": parse_html("article", (FIXTURES / "article.html").read_text()),
        "export.json": parse_json_export(
            "export",
            (FIXTURES / "export.json").read_text(),
            fields=(
                WeightedField(name="body", weight=2.0),
                WeightedField(name="keywords", weight=0.5),
            ),
        ),
        "reference.md": parse_markdown("reference", (FIXTURES / "reference.md").read_text()),
    }
    expected = json.loads((FIXTURES / "golden_parsed.json").read_text())

    assert {name: _snapshot(document) for name, document in parsed.items()} == expected
    for document in parsed.values():
        assert Chunker().chunk(document) == Chunker().chunk(document)


@given(
    paragraphs=st.lists(
        st.lists(
            st.text(
                alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=8
            ),
            min_size=1,
            max_size=80,
        ).map(" ".join),
        min_size=1,
        max_size=12,
    )
)
def test_chunks_cover_all_source_tokens_in_order(paragraphs: list[str]) -> None:
    blocks = [DocumentBlock(kind="heading", text="Generated", locator="line:1", heading_level=1)]
    blocks.extend(
        DocumentBlock(kind="paragraph", text=text, locator=f"line:{index + 2}")
        for index, text in enumerate(paragraphs)
    )
    document = ParsedDocument(
        source_id="generated", blocks=tuple(blocks), media_type="text/markdown"
    )
    tokenizer = WhitespaceTokenizer()

    chunks = Chunker(tokenizer).chunk(document)
    reconstructed: list[str] = []
    for chunk in chunks:
        reconstructed.extend(tokenizer.encode(chunk.text)[chunk.overlap_tokens :])

    expected = [token for block in document.blocks for token in tokenizer.encode(block.text)]
    assert reconstructed == expected
