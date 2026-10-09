from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from aegon_ingestion.documents import DocumentBlock, ParsedDocument

_BLOCK_TAGS = {"p", "li", "pre", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
_BOILERPLATE_TAGS = {"aside", "footer", "form", "header", "nav", "noscript", "script", "style"}
_BOILERPLATE_HINT = re.compile(
    r"(?:advert|banner|breadcrumb|cookie|footer|header|menu|nav|newsletter|promo|sidebar|social)",
    re.IGNORECASE,
)
_SPACE = re.compile(r"[ \t\f\v]+")


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str]
    children: list[object] = field(default_factory=lambda: list[object]())


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag.lower(), {name: value or "" for name, value in attrs})
        self.stack[-1].children.append(node)
        if tag.lower() not in {"br", "hr", "img", "input", "meta", "link", "source"}:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag.lower():
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        for position in range(len(self.stack) - 1, 0, -1):
            if self.stack[position].tag == tag.lower():
                del self.stack[position:]
                break

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def _is_boilerplate(node: _Node) -> bool:
    hints = " ".join(
        (node.attrs.get("class", ""), node.attrs.get("id", ""), node.attrs.get("role", ""))
    )
    return node.tag in _BOILERPLATE_TAGS or bool(_BOILERPLATE_HINT.search(hints))


def _text(node: _Node) -> str:
    pieces: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            pieces.append(child)
        elif isinstance(child, _Node) and not _is_boilerplate(child):
            pieces.append(_text(child))
    return _SPACE.sub(" ", "".join(pieces)).strip()


def _raw_text(node: _Node) -> str:
    pieces: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            pieces.append(child)
        elif isinstance(child, _Node) and not _is_boilerplate(child):
            pieces.append(_raw_text(child))
    return "".join(pieces).strip()


def _table_text(node: _Node) -> str:
    rows = _find(node, {"tr"})
    rendered: list[str] = []
    for row in rows:
        cells = _find(row, {"th", "td"})
        rendered.append(" | ".join(_text(cell) for cell in cells))
    return "\n".join(line for line in rendered if line)


def _find(node: _Node, tags: set[str]) -> list[_Node]:
    found: list[_Node] = []
    for child in node.children:
        if not isinstance(child, _Node) or _is_boilerplate(child):
            continue
        if child.tag in tags:
            found.append(child)
        else:
            found.extend(_find(child, tags))
    return found


def parse_html(source_id: str, content: str) -> ParsedDocument:
    """Extract semantic main content while dropping common page chrome."""
    parser = _TreeBuilder()
    parser.feed(content)
    candidates = _find(parser.root, {"main"}) or _find(parser.root, {"article"})
    roots = candidates or _find(parser.root, {"body"}) or [parser.root]
    nodes: list[_Node] = []
    for root in roots:
        nodes.extend(_find(root, _BLOCK_TAGS))

    blocks: list[DocumentBlock] = []
    counts: dict[str, int] = {}
    for node in nodes:
        text = (
            _raw_text(node)
            if node.tag == "pre"
            else _table_text(node)
            if node.tag == "table"
            else _text(node)
        )
        if not text:
            continue
        counts[node.tag] = counts.get(node.tag, 0) + 1
        locator = f"{node.tag}:nth-of-type({counts[node.tag]})"
        if node.tag.startswith("h"):
            blocks.append(
                DocumentBlock(
                    kind="heading",
                    text=text,
                    locator=locator,
                    heading_level=int(node.tag[1]),
                )
            )
        else:
            kind = "code" if node.tag == "pre" else "table" if node.tag == "table" else "paragraph"
            blocks.append(DocumentBlock(kind=kind, text=text, locator=locator))
    return ParsedDocument(source_id=source_id, blocks=tuple(blocks), media_type="text/html")
