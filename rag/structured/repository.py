"""Single access layer for the structured store (JSON today; swap for a DB by reimplementing this class).

File layout:
{
  "schema_version": "1.0",
  "generated_at": "...",
  "sources": [{"tool", "root_path", "files": [{"relative_path", "last_modified", "hash"}]}],
  "items": {"decisions": [...], "rules": [...], "warnings": [...], "changes": [...]}
}
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import re

from rag.structured.schema import ITEM_MODELS, ITEM_TYPES, SCHEMA_VERSION, BaseItem, StructuredQuery



def _squash(text: str) -> str:
    """lowercase, letters/digits only: 'code_review' == 'Code-Review' == 'code review'."""
    return re.sub(r"[^0-9a-z֐-׿]", "", text.lower())


class StructuredRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._items: dict[str, BaseItem] = {}
        self._files: dict[str, dict] = {}       # relative_path -> {tool, root_path, last_modified, hash}
        if self.path.is_file():
            self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            return  # incompatible store -> start fresh (everything gets re-extracted)
        for src in data.get("sources", []):
            for f in src.get("files", []):
                self._files[f["relative_path"]] = {**f, "tool": src["tool"], "root_path": src["root_path"]}
        for item_type, items in data.get("items", {}).items():
            for raw in items:
                item = ITEM_MODELS[item_type].model_validate(raw)
                self._items[item.id] = item

    def save(self) -> None:
        by_tool: dict[tuple[str, str], list[dict]] = {}
        for rel, f in sorted(self._files.items()):
            by_tool.setdefault((f["tool"], f["root_path"]), []).append(
                {"relative_path": rel, "last_modified": f["last_modified"], "hash": f["hash"]})
        data = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": [{"tool": t, "root_path": r, "files": files} for (t, r), files in sorted(by_tool.items())],
            "items": {t: [i.model_dump(mode="json") for i in self.items(t)] for t in ITEM_TYPES},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- files ----------
    def file_hash(self, relative_path: str) -> str | None:
        f = self._files.get(relative_path)
        return f["hash"] if f else None

    def replace_file_items(self, relative_path: str, file_info: dict, items: list[BaseItem]) -> None:
        """Drop every item that came from this file, then store the new ones and the file's hash."""
        self._items = {k: v for k, v in self._items.items() if v.source.relative_path != relative_path}
        for item in items:
            self._items[item.id] = item
        self._files[relative_path] = file_info

    def remove_files_not_in(self, relative_paths: set[str]) -> list[str]:
        """Forget files removed from config (and their items)."""
        gone = [p for p in self._files if p not in relative_paths]
        for p in gone:
            self.replace_file_items(p, {}, [])
            del self._files[p]
        return gone

    # ---------- items ----------
    def items(self, item_type: str | None = None) -> list[BaseItem]:
        return sorted((i for i in self._items.values() if item_type in (None, i.item_type)),
                      key=lambda i: (i.item_type, i.source.relative_path, i.source.line_range, i.id))

    def recompute_status(self) -> None:
        """Within (item_type, topic): newest observed_at is active, older ones are superseded by it."""
        groups: dict[tuple[str, str], list[BaseItem]] = {}
        for item in self._items.values():
            item.status, item.supersedes = "active", []
            groups.setdefault((item.item_type, item.topic), []).append(item)
        for group in groups.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda i: (i.observed_at, i.source.relative_path, i.source.line_range))
            newest = group[-1]
            older = [i for i in group[:-1] if i.observed_at < newest.observed_at]
            for old in older:
                old.status = "superseded"
            newest.supersedes = [o.id for o in older]

    def query(self, q: StructuredQuery, date_from: date | None = None, date_to: date | None = None) -> list[BaseItem]:
        """Run a validated StructuredQuery. Dates are passed already parsed."""
        types = set(q.item_types) or set(ITEM_TYPES)
        tags = [t.lower() for t in q.tags]
        out = []
        for item in self._items.values():
            if item.item_type not in types:
                continue
            if q.tool and item.source.tool != q.tool:
                continue
            if q.area and q.area.lower() not in item.area:
                continue
            if q.only_active and item.status != "active":
                continue
            if date_from and item.observed_at < date_from:
                continue
            if date_to and item.observed_at > date_to:
                continue
            if tags:
                hay = _squash(" ".join([*item.tags, item.topic, item.area, item.summary, item.key_text(),
                                        item.source.header, item.source.relative_path]))
                if not any(_squash(t) in hay for t in tags):
                    continue
            out.append(item)
        out.sort(key=lambda i: (i.observed_at, i.id), reverse=(q.sort != "oldest"))
        return out[: q.limit]
