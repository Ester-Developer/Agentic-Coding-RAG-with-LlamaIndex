"""LLM prompts for the structured layer (extraction, query building, structured answer)."""
from llama_index.core.prompts import PromptTemplate

EXTRACTION_PROMPT = PromptTemplate(
    """You extract structured items from ONE section of a markdown file written by/for an AI coding tool.
File: {relative_path}   Section: {header}

Extract only items that are explicitly stated in the section text. Do not invent. Empty lists are fine.
Item types and their REQUIRED type-specific fields (an item without them is discarded):
- decisions: technical/architectural choices that were made (e.g. "one shared Helm chart for all teams").
    required: "decision" = the decision itself, quoted or closely paraphrased. optional: "rationale".
- rules: instructions/guidelines someone must or should follow (e.g. "don't copy proto files", checklist items).
    required: "rule" = the instruction itself; "strength" = "must" or "should".
- warnings: sensitivities, known limitations/gaps, do-not-touch areas, stale/outdated information.
    required: "risk" = what can go wrong / what not to touch; "kind" = sensitive | limitation | do_not_touch | stale_info.
- changes: something that changed from an old value to a new value (field, value, endpoint, type, label).
    required: "subject" = what changed; "from_value" = old value; "to_value" = new value.
    Only when BOTH the old and the new value are stated. A list of modified files is NOT a change item.
Every item also needs: topic (short snake_case subject key, reuse the same key for the same subject),
summary (one English sentence), 2-5 lowercase tags, area (team/component, e.g. inbox, lens/frontend, relay/helm, ci).

Section text:
---
{section_text}
---"""
)

BUILD_QUERY_PROMPT = PromptTemplate(
    """Convert the user's question into a structured query over a store of extracted items.
Today is {today}.

Item types and their fields:
{schema}

Allowed tools: {tools}
Rules:
- item_types: only from [decisions, rules, warnings, changes]; empty list = all types.
- Relative time ("last week", "this month", "בשבוע האחרון") -> concrete ISO dates relative to today.
- Use date_from/date_to only if the question mentions time.
- tags: English keywords of the subject (translate from Hebrew), empty for "all X" questions.
- only_active=false only if the user explicitly asks about old/superseded items.
- sort: "newest" (default) or "oldest". limit: 1-{max_limit}.
- tags filter items: use ONLY concrete subject words (e.g. "proto", "kafka", "auth", "helm"). Never generic
  words like "technical", "project", "all", "important", "items" — for "all X" questions tags must be [].

Examples (assuming today is 2026-11-15):
Q: "תן לי את כל ההחלטות הטכניות"      -> {{"item_types": ["decisions"], "tags": [], "limit": {max_limit}}}
Q: "מה ההנחיה העדכנית לגבי קבצי proto?" -> {{"item_types": ["rules"], "tags": ["proto"], "only_active": true, "sort": "newest"}}
Q: "מה סומן כרגיש בשבוע האחרון?"       -> {{"item_types": ["warnings"], "tags": [], "date_from": "2026-11-08", "date_to": "2026-11-15"}}
Q: "אילו שינויים נעשו ב-API של ההתחברות?" -> {{"item_types": ["changes"], "tags": ["auth", "login"]}}
{feedback}
Question: {question}"""
)

STRUCTURED_ANSWER_PROMPT = """You answer using ONLY the numbered items below (extracted from project markdown files).
Rules:
- Answer in {language}.
- Use only these items. Do not add general knowledge.
- If the items do not answer the question, reply with exactly NO_ANSWER and nothing else.
- If items contradict each other, state the contradiction and cite both.
- Cite every claim with the item number in square brackets, e.g. [1] or [2][3].
- For list questions, include every relevant item.

Items:
{items}

Question: {question}
Answer:"""
