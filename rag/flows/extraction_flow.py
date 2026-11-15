"""Extraction Workflow: md files -> structured items (decisions/rules/warnings/changes) in the repository.

load_sources -> split_sections -> (fan-out) extract_section [LLM] -> validate_items (collect)
-> merge_items -> save_items.  Only extract_section calls the LLM.
Files whose hash didn't change since the last run are not re-extracted.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from llama_index.core import Document
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, Field, ValidationError

from rag.flows.logging_utils import emit, log_enter
from rag.loader import DEFAULT_CONFIG, ConfigError, load_config, load_documents
from rag.structured.prompts import EXTRACTION_PROMPT
from rag.structured.repository import StructuredRepository
from rag.structured.schema import ITEM_MODELS, ITEM_TYPES, BaseItem, SectionExtraction, SourceRef, make_id
from rag.structured.sections import Section, split_sections

logger = logging.getLogger("rag.flows.extraction")

_ENUM_DEFAULTS = {"rules": ("strength", {"must", "should"}, "should"),
                  "warnings": ("kind", {"sensitive", "limitation", "do_not_touch", "stale_info"}, "limitation")}


# ---------- Events ----------
class ExtractStart(StartEvent):
    config_path: str = str(DEFAULT_CONFIG)
    force: bool = False                       # re-extract even unchanged files


class FilesToExtract(Event):
    documents: list[Document]                 # only new/changed files


class SectionToExtract(Event):
    section: Section


class SectionExtracted(Event):
    section: Section
    extraction: SectionExtraction | None = None
    error: str = ""


class ItemsValidated(Event):
    records: list[BaseItem]


class ItemsMerged(Event):
    items_by_file: dict[str, list[BaseItem]]


class ExtractSummary(BaseModel):
    ok: bool
    extracted_files: list[str] = Field(default_factory=list)
    unchanged_files: list[str] = Field(default_factory=list)
    sections: int = 0
    items_by_type: dict[str, int] = Field(default_factory=dict)
    items_by_tool: dict[str, int] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)
    duplicates_merged: int = 0
    failed_files: list[str] = Field(default_factory=list)
    error: str = ""


# ---------- State ----------
class ExtractState(BaseModel):
    config: dict = Field(default_factory=dict)
    file_infos: dict[str, dict] = Field(default_factory=dict)   # rel_path -> hash/mtime/tool, saved at the end
    unchanged: list[str] = Field(default_factory=list)          # for the summary
    expected_sections: int = 0                                  # how many SectionExtracted to collect
    skipped: list[str] = Field(default_factory=list)            # every rejected item/section, with reason
    duplicates_merged: int = 0
    failed_files: list[str] = Field(default_factory=list)       # files with a failed section -> hash not saved, retried next run


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_to_item(item_type: str, raw: dict, section: Section) -> BaseItem:
    """Build a strict item from one raw LLM item + section info. Raises ValidationError if invalid."""
    data = {k: v for k, v in raw.items() if v is not None}
    if item_type in _ENUM_DEFAULTS:
        field, allowed, default = _ENUM_DEFAULTS[item_type]
        value = str(data.get(field) or "").strip().lower()
        data[field] = value if value in allowed else default
    data.update(
        source=SourceRef(tool=section.tool, relative_path=section.relative_path,
                         header=section.header, line_range=section.line_range),
        observed_at=section.observed_at, date_source=section.date_source)
    item = ITEM_MODELS[item_type].model_validate(data)
    item.id = make_id(item)
    return item


def convert_extractions(events: list[SectionExtracted]) -> tuple[list[BaseItem], list[str]]:
    """Raw LLM items -> strict items. Returns (valid items, skip reasons). Pure: no LLM, no IO."""
    items: list[BaseItem] = []
    skipped: list[str] = []
    for e in events:
        where = f"{e.section.relative_path} '{e.section.header}'"
        if e.extraction is None:
            skipped.append(f"{where}: section extraction failed ({e.error[:120]})")
            continue
        for item_type in ITEM_TYPES:
            for raw in getattr(e.extraction, item_type):
                try:
                    items.append(raw_to_item(item_type, raw.model_dump(), e.section))
                except ValidationError as err:
                    bad = ", ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in err.errors())
                    skipped.append(f"{where}: invalid {item_type} item ({bad})")
    return items, skipped


class ExtractionWorkflow(Workflow):
    def __init__(self, llm=None, repository: StructuredRepository | None = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm
        self._repo = repository

    def _get_llm(self, cfg: dict):
        if self._llm is None:
            from rag.llm import get_llm

            self._llm = get_llm(cfg, temperature=0.0)
        return self._llm

    def _get_repo(self, cfg: dict) -> StructuredRepository:
        if self._repo is None:
            self._repo = StructuredRepository(Path(DEFAULT_CONFIG).parent / cfg["structured"]["store_path"])
        return self._repo

    @step
    async def load_sources(self, ctx: Context[ExtractState], ev: ExtractStart) -> FilesToExtract | StopEvent:
        """Receives: ExtractStart(config_path, force).
        Loads files (stage-1 loader), compares each file's sha256 with the repository.
        Returns: FilesToExtract(documents) with only new/changed files.
        May emit: StopEvent(ok=False) on config/missing-file error; StopEvent(ok=True) if nothing changed."""
        log_enter(logger, "load_sources", ev)
        try:
            cfg = load_config(ev.config_path)
            docs = load_documents(ev.config_path)
        except (FileNotFoundError, ConfigError) as e:
            return emit(logger, "load_sources", StopEvent(result=ExtractSummary(ok=False, error=str(e))), "load failed")
        repo = self._get_repo(cfg)
        gone = repo.remove_files_not_in({d.metadata["relative_path"] for d in docs})
        if gone:
            logger.info("[load_sources] removed files no longer in config: %s", gone)

        changed, unchanged, infos = [], [], {}
        for d in docs:
            rel, h = d.metadata["relative_path"], file_sha256(d.metadata["file_path"])
            infos[rel] = {"tool": d.metadata["tool"], "root_path": cfg["source_root"],
                          "last_modified": d.metadata["last_modified_date"], "hash": h}
            if not ev.force and repo.file_hash(rel) == h:
                unchanged.append(rel)
            else:
                changed.append(d)
        async with ctx.store.edit_state() as s:
            s.config, s.file_infos, s.unchanged = cfg, infos, unchanged
        if not changed:
            repo.recompute_status()
            repo.save()
            summary = ExtractSummary(ok=True, unchanged_files=unchanged)
            return emit(logger, "load_sources", StopEvent(result=summary), "all files unchanged -> nothing to extract")
        return emit(logger, "load_sources", FilesToExtract(documents=changed),
                    f"{len(changed)} changed, {len(unchanged)} unchanged")

    @step
    async def split_into_sections(self, ctx: Context[ExtractState], ev: FilesToExtract) -> SectionToExtract | None:
        """Receives: FilesToExtract.
        Splits each file by md headers, with exact line ranges and observed_at/date_source.
        Emits: one SectionToExtract per section (fan-out via ctx.send_event); sections shorter than
        `min_section_chars` (header-only) are skipped with a warning. Returns None."""
        log_enter(logger, "split_into_sections", ev)
        min_chars = (await ctx.store.get_state()).config["structured"]["min_section_chars"]
        to_send, skipped = [], []
        for doc in ev.documents:
            for sec in split_sections(doc):
                body = sec.text.split("\n", 1)[1].strip() if "\n" in sec.text else ""
                if len(body) < min_chars:
                    skipped.append(f"{sec.relative_path} '{sec.header}': section too short, not extracted")
                else:
                    to_send.append(sec)
        for s in skipped:
            logger.warning("[split_into_sections] %s", s)
        async with ctx.store.edit_state() as st:
            st.expected_sections = len(to_send)
            st.skipped.extend(skipped)
        if not to_send:
            ctx.send_event(ItemsValidated(records=[]))
        for sec in to_send:
            ctx.send_event(SectionToExtract(section=sec))
        logger.info("[split_into_sections] exit -> %d x SectionToExtract (one per section)", len(to_send))
        return None

    @step(num_workers=2)
    async def extract_section(self, ctx: Context[ExtractState], ev: SectionToExtract) -> SectionExtracted:
        """Receives: SectionToExtract.  [LLM]
        Structured extraction (llm.astructured_predict -> SectionExtraction) for one section.
        Returns: SectionExtracted(section, extraction) — or with `error` set if the LLM call failed
        after retries (never raises, so the collector always gets every section)."""
        log_enter(logger, "extract_section", ev)
        cfg = (await ctx.store.get_state()).config
        sec, last_err = ev.section, ""
        for attempt in range(4):
            try:
                result = await self._get_llm(cfg).astructured_predict(
                    SectionExtraction, EXTRACTION_PROMPT,
                    relative_path=sec.relative_path, header=sec.header, section_text=sec.text)
                n = sum(len(getattr(result, t)) for t in ITEM_TYPES)
                return emit(logger, "extract_section", SectionExtracted(section=sec, extraction=result),
                            f"'{sec.header}': {n} raw items")
            except Exception as e:   # LLM/format/network error -> retry, then report
                last_err = f"{type(e).__name__}: {e}"
                logger.warning("[extract_section] '%s' attempt %d failed: %s", sec.header, attempt + 1, last_err[:200])
                await asyncio.sleep(6 * 2 ** attempt)   # 6s, 12s, 24s: room for rate limits (429)
        return emit(logger, "extract_section", SectionExtracted(section=sec, error=last_err),
                    f"'{sec.header}': extraction failed")

    @step
    async def validate_items(self, ctx: Context[ExtractState], ev: SectionExtracted) -> ItemsValidated | None:
        """Receives: SectionExtracted (collects all of them before continuing).
        Turns raw items into strict Pydantic items; invalid items (no summary, bad fields) are skipped
        with a warning. Returns: ItemsValidated(items) once every section has arrived; None while waiting."""
        state = await ctx.store.get_state()
        events = ctx.collect_events(ev, [SectionExtracted] * state.expected_sections)
        if events is None:
            return None
        log_enter(logger, "validate_items", ev)
        items, skipped = convert_extractions(events)
        failed = sorted({e.section.relative_path for e in events if e.extraction is None})
        for s in skipped:
            logger.warning("[validate_items] skipping %s", s)
        async with ctx.store.edit_state() as st:
            st.skipped.extend(skipped)
            st.failed_files = failed
        return emit(logger, "validate_items", ItemsValidated(records=items),
                    f"{len(items)} valid, {len(skipped)} skipped")

    @step
    async def merge_items(self, ctx: Context[ExtractState], ev: ItemsValidated) -> ItemsMerged:
        """Receives: ItemsValidated.
        Merges duplicates (same stable id, e.g. the same item found in two sections): tags are united,
        the first occurrence's source is kept. Returns: ItemsMerged(items_by_file). No LLM."""
        log_enter(logger, "merge_items", ev)
        merged: dict[str, BaseItem] = {}
        dups = 0
        for item in ev.records:
            if item.id in merged:
                dups += 1
                merged[item.id].tags = sorted(set(merged[item.id].tags) | set(item.tags))
            else:
                merged[item.id] = item
        by_file: dict[str, list[BaseItem]] = {}
        for item in merged.values():
            by_file.setdefault(item.source.relative_path, []).append(item)
        async with ctx.store.edit_state() as st:
            st.duplicates_merged += dups
        return emit(logger, "merge_items", ItemsMerged(items_by_file=by_file),
                    f"{len(merged)} unique items, {dups} duplicates merged")

    @step
    async def save_items(self, ctx: Context[ExtractState], ev: ItemsMerged) -> StopEvent:
        """Receives: ItemsMerged.
        Replaces the items of every re-extracted file in the repository, recomputes active/superseded,
        saves. Returns: StopEvent(ExtractSummary). No LLM."""
        log_enter(logger, "save_items", ev)
        state = await ctx.store.get_state()
        repo = self._get_repo(state.config)
        extracted = [rel for rel in state.file_infos if rel not in state.unchanged]
        for rel in extracted:
            info = dict(state.file_infos[rel])
            if rel in state.failed_files:      # keep what we got, but force a retry next run
                info["hash"] = ""
                logger.warning("[save_items] %s had failed sections; it will be re-extracted next run", rel)
            repo.replace_file_items(rel, info, ev.items_by_file.get(rel, []))
        repo.recompute_status()
        repo.save()
        all_items = repo.items()
        by_type = {t: sum(1 for i in all_items if i.item_type == t) for t in ITEM_TYPES}
        by_tool: dict[str, int] = {}
        for i in all_items:
            by_tool[i.source.tool] = by_tool.get(i.source.tool, 0) + 1
        summary = ExtractSummary(ok=True, extracted_files=extracted, unchanged_files=state.unchanged,
                                 sections=state.expected_sections, items_by_type=by_type, items_by_tool=by_tool,
                                 skipped=state.skipped, duplicates_merged=state.duplicates_merged,
                                 failed_files=state.failed_files)
        return emit(logger, "save_items", StopEvent(result=summary), f"{len(all_items)} items in store")


async def run_extraction(config_path: str | Path = DEFAULT_CONFIG, force: bool = False, **deps) -> ExtractSummary:
    wf = ExtractionWorkflow(timeout=1800, **deps)
    return await wf.run(start_event=ExtractStart(config_path=str(config_path), force=force))
