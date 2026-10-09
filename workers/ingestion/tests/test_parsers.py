import json

import pytest
from aegon_ingestion.parsers import WeightedField, parse_html, parse_json_export, parse_markdown
from pydantic import ValidationError


def test_markdown_keeps_code_and_table_blocks_intact() -> None:
    document = parse_markdown(
        "guide",
        "# Install\n\nUse this.\n\n```py\nprint('ok')\n```\n\n| A | B |\n| - | - |\n| 1 | 2 |",
    )

    assert [block.kind for block in document.blocks] == ["heading", "paragraph", "code", "table"]
    assert document.blocks[2].text == "```py\nprint('ok')\n```"
    assert document.blocks[3].text.endswith("| 1 | 2 |")


def test_mdx_strips_components_and_keeps_text_exports() -> None:
    document = parse_markdown(
        "page",
        'import Card from "./Card"\n\n'
        'export const summary = "Visible summary";\n\n'
        "<Card>Useful text</Card>",
        mdx=True,
    )

    assert [block.text for block in document.blocks] == ["Visible summary", "Useful text"]


def test_html_prefers_main_and_removes_boilerplate() -> None:
    document = parse_html(
        "article",
        "<body><nav>Menu</nav><main><h1>Title</h1><p>Evidence text.</p>"
        "<aside>Promotion</aside></main><footer>Legal</footer></body>",
    )

    assert [block.text for block in document.blocks] == ["Title", "Evidence text."]


def test_json_export_validates_schema_and_applies_weights() -> None:
    payload = json.dumps(
        {
            "schema_version": "1",
            "documents": [{"id": "1", "title": "Runbook", "fields": {"body": "Restart it"}}],
        }
    )
    document = parse_json_export(
        "export",
        payload,
        fields=(WeightedField(name="body", weight=2.0),),
    )

    assert document.blocks[1].weight == 2.0
    assert document.blocks[1].locator == "/documents/0/fields/body"

    with pytest.raises(ValidationError):
        parse_json_export(
            "export",
            '{"schema_version":"1","documents":['
            '{"id":"1","title":"Bad","fields":{},"tenant":"x"}]}',
            fields=(),
        )


def test_html_preserves_preformatted_code_and_table_cells() -> None:
    document = parse_html(
        "reference",
        "<main><pre>if ready:\n    run()</pre>"
        "<table><tr><th>Name</th><th>State</th></tr><tr><td>A</td><td>ready</td></tr></table></main>",
    )

    assert document.blocks[0].text == "if ready:\n    run()"
    assert document.blocks[1].text == "Name | State\nA | ready"
