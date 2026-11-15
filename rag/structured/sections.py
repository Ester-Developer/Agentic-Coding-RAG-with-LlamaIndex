"""Split a loaded Document into header sections with exact line ranges, and find its content date."""
from __future__ import annotations

import re
from datetime import date, datetime

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from pydantic import BaseModel

# "**Date:** October 20, 2026" / "Date: 2026-10-20" near the top of a file
_DATE_LINE = re.compile(r"^\s*(?:\*\*)?(?:Date|Updated|Last updated)\s*:?\s*(?:\*\*)?\s*:?\s*(.+?)\s*$",
                        re.IGNORECASE | re.MULTILINE)
_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%B %d %Y")


class Section(BaseModel):
    tool: str
    relative_path: str
    header: str
    line_range: tuple[int, int]
    text: str
    observed_at: date
    date_source: str          # "content" | "file_mtime"


def content_date(text: str, head_lines: int = 30) -> date | None:
    """A date written in the file itself (only the first lines, where metadata lives)."""
    head = "\n".join(text.splitlines()[:head_lines])
    for m in _DATE_LINE.finditer(head):
        raw = m.group(1).strip().strip("*").strip()
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
    return None


def split_sections(doc: Document) -> list[Section]:
    md = doc.metadata
    found = content_date(doc.text)
    observed_at, date_source = (found, "content") if found else (
        date.fromisoformat(md["last_modified_date"]), "file_mtime")

    sections, cursor = [], 0
    for node in MarkdownNodeParser().get_nodes_from_documents([Document(text=doc.text)]):
        text = node.get_content()
        start = doc.text.find(text, cursor)
        if start < 0:                       # header re-rendered differently: fall back to its first line
            start = doc.text.find(text.split("\n", 1)[0], cursor)
        start = max(start, 0)
        cursor = start + 1
        first = doc.text.count("\n", 0, start) + 1
        header = text.split("\n", 1)[0].lstrip("#").strip() if text.startswith("#") else ""
        sections.append(Section(
            tool=md["tool"], relative_path=md["relative_path"], header=header,
            line_range=(first, first + text.count("\n")), text=text,
            observed_at=observed_at, date_source=date_source))
    return sections
