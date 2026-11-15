"""Schema for the structured data layer.

Two layers of models:
- Raw*  : what the LLM returns. Every field optional, so one bad item can't fail a whole section.
- Item models (Decision, Rule, Warning, Change): strict, what we store. Built in code from Raw* +
  section info; an item that fails validation here is skipped with a warning.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"
ITEM_TYPES = ("decisions", "rules", "warnings", "changes")
ItemType = Literal["decisions", "rules", "warnings", "changes"]


# ---------- shared parts ----------
class SourceRef(BaseModel):
    tool: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    header: str = ""                     # section title the item came from
    line_range: tuple[int, int]          # 1-based, inclusive, in the source file


class BaseItem(BaseModel):
    id: str = ""                         # stable hash, set by make_id()
    item_type: ItemType
    topic: str = Field(min_length=1)     # normalized subject key; groups items for supersede logic
    summary: str = Field(min_length=5)
    tags: list[str] = Field(default_factory=list)
    area: str = ""                       # team/component the item is about (e.g. "inbox", "lens/frontend")
    source: SourceRef
    observed_at: date
    date_source: Literal["content", "file_mtime"]
    status: Literal["active", "superseded"] = "active"
    supersedes: list[str] = Field(default_factory=list)   # ids of older items this one replaces

    @field_validator("tags")
    @classmethod
    def _norm_tags(cls, v: list[str]) -> list[str]:
        return sorted({t.strip().lower() for t in v if t and t.strip()})

    @field_validator("topic", "area")
    @classmethod
    def _norm_key(cls, v: str) -> str:
        return re.sub(r"[^a-z0-9/]+", "_", (v or "").strip().lower()).strip("_")

    def key_text(self) -> str:
        """The item's main content, used in the stable id."""
        raise NotImplementedError


class Decision(BaseItem):
    item_type: Literal["decisions"] = "decisions"
    decision: str = Field(min_length=3)          # what was decided
    rationale: str = ""                           # why, if stated

    def key_text(self) -> str:
        return self.decision


class Rule(BaseItem):
    item_type: Literal["rules"] = "rules"
    rule: str = Field(min_length=3)              # the instruction itself
    strength: Literal["must", "should"] = "should"

    def key_text(self) -> str:
        return self.rule


class WarningItem(BaseItem):
    item_type: Literal["warnings"] = "warnings"
    risk: str = Field(min_length=3)              # what can go wrong / what not to touch
    kind: Literal["sensitive", "limitation", "do_not_touch", "stale_info"] = "limitation"

    def key_text(self) -> str:
        return self.risk


class Change(BaseItem):
    item_type: Literal["changes"] = "changes"
    subject: str = Field(min_length=1)           # field / value / endpoint that changed
    from_value: str = Field(min_length=1)
    to_value: str = Field(min_length=1)

    def key_text(self) -> str:
        return f"{self.subject}:{self.from_value}->{self.to_value}"


ITEM_MODELS: dict[str, type[BaseItem]] = {
    "decisions": Decision, "rules": Rule, "warnings": WarningItem, "changes": Change,
}


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def make_id(item: BaseItem) -> str:
    """Stable id = hash(type + source file + topic + main content).
    Header/lines are NOT part of it, so the same item found in two sections of a file merges."""
    raw = "|".join([item.item_type, item.source.relative_path, item.topic, _norm(item.key_text())])
    return f"{item.item_type[:3]}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


# ---------- what the LLM returns (lenient) ----------
class _RawCommon(BaseModel):
    topic: Optional[str] = Field(None, description="short snake_case subject key, e.g. 'auth_login_endpoint', 'kafka_topic_layout'")
    summary: Optional[str] = Field(None, description="one sentence in English")
    tags: list[str] = Field(default_factory=list, description="2-5 lowercase keywords")
    area: Optional[str] = Field(None, description="team/component, e.g. 'inbox', 'lens/frontend', 'relay/helm', 'ci'")


class RawDecision(_RawCommon):
    decision: Optional[str] = Field(None, description="REQUIRED: the decision itself")
    rationale: Optional[str] = Field(None, description="why, if stated")


class RawRule(_RawCommon):
    rule: Optional[str] = Field(None, description="REQUIRED: the instruction itself")
    strength: Optional[str] = Field(None, description="REQUIRED: 'must' or 'should'")


class RawWarning(_RawCommon):
    risk: Optional[str] = Field(None, description="REQUIRED: what can go wrong / what not to touch")
    kind: Optional[str] = Field(None, description="one of: sensitive, limitation, do_not_touch, stale_info")


class RawChange(_RawCommon):
    subject: Optional[str] = Field(None, description="REQUIRED: what changed (field/value/endpoint)")
    from_value: Optional[str] = Field(None, description="REQUIRED: old value")
    to_value: Optional[str] = Field(None, description="REQUIRED: new value")


class SectionExtraction(BaseModel):
    """Items found in one markdown section. Empty lists are fine."""
    decisions: list[RawDecision] = Field(default_factory=list, description="technical/architectural decisions")
    rules: list[RawRule] = Field(default_factory=list, description="work rules and instructions to follow")
    warnings: list[RawWarning] = Field(default_factory=list, description="sensitivities, known limitations, do-not-touch, stale info")
    changes: list[RawChange] = Field(default_factory=list, description="a field/value/endpoint that changed from X to Y")


# ---------- structured query (what the LLM builds; validated in code) ----------
class StructuredQuery(BaseModel):
    """Query over the structured store. No free code/SQL — only these fields."""
    item_types: list[str] = Field(default_factory=list, description="subset of: decisions, rules, warnings, changes. Empty = all")
    tags: list[str] = Field(default_factory=list, description="keywords to match against item tags/topic/area")
    tool: Optional[str] = Field(None, description="claude_code or kiro, only if the question names a tool")
    area: Optional[str] = Field(None, description="team/component if the question names one")
    date_from: Optional[str] = Field(None, description="ISO date YYYY-MM-DD, inclusive")
    date_to: Optional[str] = Field(None, description="ISO date YYYY-MM-DD, inclusive")
    only_active: bool = Field(True, description="false only if the user asks about old/superseded items")
    sort: str = Field("newest", description="'newest' or 'oldest'")
    limit: int = Field(20, description="max items to return")
