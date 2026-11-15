"""Extraction: valid item, invalid item, duplicates, re-run skips unchanged files, sections/dates."""
from datetime import date

from llama_index.core import Document
from llama_index.core.workflow import Context

from rag.flows.extraction_flow import (ExtractionWorkflow, ItemsValidated, SectionExtracted,
                                      convert_extractions, raw_to_item)
from rag.structured.repository import StructuredRepository
from rag.structured.schema import RawChange, RawDecision, RawRule, SectionExtraction
from rag.structured.sections import Section, content_date, split_sections
from tests.conftest import FakeLLM, requires_sources

SEC = Section(tool="kiro", relative_path=".kiro/skills/code-review.md", header="Security", line_range=(22, 25),
              text="### Security\n- No sensitive data in logs", observed_at=date(2026, 9, 9), date_source="file_mtime")


def test_valid_item_gets_source_and_stable_id():
    raw = RawRule(topic="Sensitive Logs", summary="Never log sensitive data.", tags=["Security"], rule="No sensitive data in logs",
                  strength="MUST")
    item = raw_to_item("rules", raw.model_dump(), SEC)
    assert item.source.line_range == (22, 25) and item.topic == "sensitive_logs" and item.strength == "must"
    assert item.id == raw_to_item("rules", raw.model_dump(), SEC.model_copy(update={"header": "Other"})).id


def test_invalid_items_skipped_not_crash():
    extraction = SectionExtraction(
        rules=[RawRule(topic="t", summary=None, rule="x y z"),                       # no summary -> rejected
               RawRule(topic="ok", summary="A valid rule here.", rule="do it")],
        changes=[RawChange(topic="c", summary="Changed something.", subject="field", from_value=None, to_value="b")])
    items, skipped = convert_extractions([SectionExtracted(section=SEC, extraction=extraction)])
    assert len(items) == 1 and items[0].rule == "do it"
    assert len(skipped) == 2 and "summary" in skipped[0] and "from_value" in skipped[1]


def test_failed_section_is_reported():
    items, skipped = convert_extractions([SectionExtracted(section=SEC, error="timeout")])
    assert items == [] and "extraction failed" in skipped[0]


async def test_duplicates_merged_by_id():
    w = ExtractionWorkflow(llm=FakeLLM())
    ctx = Context(w)
    raw = RawDecision(topic="kafka", summary="One topic for all.", tags=["kafka"], decision="single topic")
    a = raw_to_item("decisions", raw.model_dump(), SEC)
    b = raw_to_item("decisions", {**raw.model_dump(), "tags": ["bus"]}, SEC.model_copy(update={"header": "Other"}))
    out = await w.merge_items(ctx, ItemsValidated(records=[a, b]))
    merged = out.items_by_file[SEC.relative_path]
    assert len(merged) == 1 and merged[0].tags == ["bus", "kafka"]
    assert (await ctx.store.get_state()).duplicates_merged == 1


def test_content_date_and_line_ranges():
    text = "# Title\n\n**Date:** October 20, 2026  \n\n## Part A\nline a\n\n## Part B\nline b\n"
    assert content_date(text) == date(2026, 10, 20)
    doc = Document(text=text, metadata={"tool": "claude_code", "relative_path": "x.md", "last_modified_date": "2026-11-01"})
    secs = split_sections(doc)
    part_b = [s for s in secs if s.header == "Part B"][0]
    assert part_b.line_range == (8, 9) and part_b.date_source == "content"


@requires_sources
async def test_rerun_skips_unchanged_files(cfg, tmp_path):
    repo = StructuredRepository(tmp_path / "s.json")
    llm = FakeLLM(structured=[SectionExtraction()] * 200)
    first = await ExtractionWorkflow(llm=llm, repository=repo, timeout=60).run()
    calls_after_first = len(llm.structured_calls)
    second = await ExtractionWorkflow(llm=llm, repository=repo, timeout=60).run()
    assert first.ok and calls_after_first > 0
    assert second.extracted_files == [] and len(second.unchanged_files) == 3
    assert len(llm.structured_calls) == calls_after_first   # no new LLM calls


def test_supersede_newest_wins(repo):
    from rag.structured.schema import Rule, SourceRef, make_id
    newer = Rule(topic="proto_files", summary="Proto files now live in a registry.", rule="use the registry",
                 source=SourceRef(tool="kiro", relative_path="new.md", line_range=(1, 2)),
                 observed_at=date(2026, 11, 1), date_source="content")
    newer.id = make_id(newer)
    repo.replace_file_items("new.md", {"tool": "kiro", "root_path": "X", "last_modified": "", "hash": "h"}, [newer])
    repo.recompute_status()
    rules = {r.source.relative_path: r for r in repo.items("rules")}
    assert rules["CLAUDE.md"].status == "superseded" and rules["new.md"].supersedes == [rules["CLAUDE.md"].id]


@requires_sources
async def test_file_with_failed_section_is_retried_next_run(cfg, tmp_path, monkeypatch):
    import rag.flows.extraction_flow as ef

    async def no_sleep(_):
        return None
    monkeypatch.setattr(ef.asyncio, "sleep", no_sleep)
    repo = StructuredRepository(tmp_path / "s.json")
    failing = FakeLLM(structured=[RuntimeError("429")] * 1000)
    first = await ExtractionWorkflow(llm=failing, repository=repo, timeout=60).run()
    assert first.ok and len(first.failed_files) == 3
    ok_llm = FakeLLM(structured=[SectionExtraction()] * 200)
    second = await ExtractionWorkflow(llm=ok_llm, repository=repo, timeout=60).run()
    assert len(second.extracted_files) == 3 and second.failed_files == []   # retried, not "unchanged"
