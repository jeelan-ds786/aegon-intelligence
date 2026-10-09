import re

from aegon_ingestion.documents import DocumentBlock, ParsedDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_DIVIDER = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")
_IMPORT_EXPORT = re.compile(r"^\s*(?:import\s|export\s+(?!const\s+\w+\s*=\s*['\"]))")
_TEXT_EXPORT = re.compile(r"^\s*export\s+const\s+\w+\s*=\s*(['\"])(.*?)\1\s*;?\s*$")
_JSX_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_JSX_EXPRESSION = re.compile(r"\{[^{}]*\}")


def _strip_mdx_syntax(line: str) -> str:
    exported_text = _TEXT_EXPORT.match(line)
    if exported_text:
        return exported_text.group(2)
    if _IMPORT_EXPORT.match(line):
        return ""
    return _JSX_EXPRESSION.sub("", _JSX_TAG.sub("", line)).strip()


def parse_markdown(source_id: str, content: str, *, mdx: bool = False) -> ParsedDocument:
    """Parse Markdown or MDX into source-aligned semantic blocks."""
    lines = content.splitlines()
    blocks: list[DocumentBlock] = []
    index = 0
    if lines and lines[0].strip() == "---":
        index = 1
        while index < len(lines) and lines[index].strip() != "---":
            index += 1
        index = min(index + 1, len(lines))

    while index < len(lines):
        raw_line = lines[index]
        line = _strip_mdx_syntax(raw_line) if mdx else raw_line
        if not line.strip():
            index += 1
            continue
        start = index + 1
        heading = _HEADING.match(line)
        if heading:
            blocks.append(
                DocumentBlock(
                    kind="heading",
                    text=heading.group(2).strip(),
                    locator=f"line:{start}",
                    heading_level=len(heading.group(1)),
                )
            )
            index += 1
            continue
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            code_lines = [line]
            index += 1
            while index < len(lines):
                code_line = lines[index]
                code_lines.append(code_line)
                index += 1
                if code_line.strip().startswith(marker):
                    break
            blocks.append(
                DocumentBlock(
                    kind="code",
                    text="\n".join(code_lines),
                    locator=f"lines:{start}-{index}",
                )
            )
            continue
        if index + 1 < len(lines) and "|" in line and _TABLE_DIVIDER.match(lines[index + 1]):
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            blocks.append(
                DocumentBlock(
                    kind="table",
                    text="\n".join(table_lines),
                    locator=f"lines:{start}-{index}",
                )
            )
            continue

        paragraph = [line.strip()]
        index += 1
        while index < len(lines):
            next_raw = lines[index]
            next_line = _strip_mdx_syntax(next_raw) if mdx else next_raw
            if not next_line.strip() or _HEADING.match(next_line) or _FENCE.match(next_line):
                break
            if (
                index + 1 < len(lines)
                and "|" in next_line
                and _TABLE_DIVIDER.match(lines[index + 1])
            ):
                break
            paragraph.append(next_line.strip())
            index += 1
        text = "\n".join(part for part in paragraph if part)
        if text:
            blocks.append(
                DocumentBlock(kind="paragraph", text=text, locator=f"lines:{start}-{index}")
            )

    media_type = "text/mdx" if mdx else "text/markdown"
    return ParsedDocument(source_id=source_id, blocks=tuple(blocks), media_type=media_type)
