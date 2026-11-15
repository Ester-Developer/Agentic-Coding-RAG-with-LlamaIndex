# RAG -02 — RAG מעל קבצי md של כלי Agentic Coding

מערכת RAG מעל קבצי md ש-Claude Code ו-Kiro יצרו בפרויקט אחר (Resonance-Flow).
הכול בנוי כ-**LlamaIndex Workflows** מבוססי אירועים:

| Workflow | קובץ | מה הוא עושה |
|---|---|---|
| Ingestion | `rag/flows/ingestion_flow.py` | md → chunks → Cohere embeddings → Pinecone |
| Extraction | `rag/flows/extraction_flow.py` | md → פריטים מובנים (decisions / rules / warnings / changes) → JSON |
| Query | `rag/flows/query_flow.py` | שאלה → Router → מסלול סמנטי או מובנה → תשובה עם מקורות |

פרויקט המקור (`source_root` ב-`config.yaml`) הוא **לקריאה בלבד**. הקבצים נקראים במקומם, ואף פעם לא מועתקים או משתנים.

## מבנה
| קובץ | תפקיד |
|---|---|
| `config.yaml` | כל ההגדרות והספים: קבצים, chunking, מודלים, Pinecone, ספי שאילתה, שכבה מובנית |
| `prompts/routes.yaml` | תיאורי המסלולים לראוטר, עם דוגמאות לשאלות |
| `rag/loader.py`, `rag/chunking.py`, `rag/embedding.py`, `rag/vector_store.py` | אבני הבניין (שלב 1–5) |
| `rag/structured/schema.py` | סכמת הפריטים (Pydantic) + `StructuredQuery` |
| `rag/structured/repository.py` | שכבת הגישה היחידה לאחסון המובנה |
| `rag/structured/sections.py` | פיצול לסעיפים עם line_range ותאריך מהתוכן |
| `rag/structured/prompts.py` | prompts לחילוץ, לבניית שאילתה ולתשובה מובנית |
| `ingest.py`, `extract.py`, `app.py` | הרצת Ingestion / Extraction / ממשק Gradio |
| `eval_router.py`, `eval/router_eval.yaml` | eval של הראוטר |
| `visualize.py` → `docs/workflow_*.html` | תרשימי Workflow |
| `tests/` | pytest עם mocks ל-Cohere, Pinecone ול-LLM |

## התקנה והרצה
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # למלא COHERE_API_KEY ו-PINECONE_API_KEY

python ingest.py              # אינדוקס ל-Pinecone (ריצה חוזרת בטוחה, בלי כפילויות)
python extract.py             # חילוץ מובנה ל-data/structured_store.json (קבצים שלא השתנו מדולגים; --force לחילוץ מלא)
python app.py                 # ממשק Gradio
python -m pytest              # בדיקות (בלי רשת)
python eval_router.py         # eval לראוטר (--full מריץ גם את ה-Workflow המלא)
python visualize.py           # docs/workflow_*.html
```

## תרשימי זרימה

### Ingestion
```
IngestStart ─► load ─► DocumentsLoaded ─► validate_documents ─► DocumentsValidated ─► chunk ─► NodesChunked
  ─► validate_nodes ─► NodesValidated ─► prepare_index ─► IndexReady ─► embed ─► NodesEmbedded ─► store ─► StopEvent
(קובץ חסר / אין מסמכים תקינים / אי-התאמת מימד → StopEvent(ok=False); מסמך או Node פסולים → דילוג + warning)
```

### Extraction
```
ExtractStart ─► load_sources ─┬─► StopEvent (כל הקבצים לא השתנו)
                              └─► FilesToExtract ─► split_into_sections ─► N × SectionToExtract   (fan-out)
  ─► extract_section [LLM, 2 workers] ─► SectionExtracted ─► validate_items (collect N) ─► ItemsValidated
  ─► merge_items ─► ItemsMerged ─► save_items ─► StopEvent(ExtractSummary)
```

### Query
```
QueryStart ─► validate_query ─┬─► StopEvent (ריקה/קצרה)
                              └─► RouteRequested ─► route_query [Selector/LLM]
                                     ├─► SemanticSearchRequested ───────────────────────────────┐
                                     └─► StructuredQueryRequested                               │
                                            ─► build_structured_query [LLM] ◄─┐                 │
                                            ─► StructuredQueryBuilt           │ StructuredQueryInvalid (בנייה מחדש אחת)
                                            ─► validate_structured_query ─────┤                 │
                                                  └─► StructuredQueryValid    └─► FallbackRequested ─► start_fallback ─┤
                                            ─► execute_structured_query ─┬─► FallbackRequested (אין תוצאות) ───────────┤
                                                                         └─► StructuredResultsFound                    │
                                            ─► synthesize_structured [LLM] ─► StructuredAnswerDrafted                  │
                                            ─► validate_structured_answer ─► StopEvent                                 │
                                                                                                                       │
  מסלול סמנטי:  retrieve ◄──────────────────────────────────────────────────────────────────────────────────────────────┘
     ─► RetrievalDone ─► evaluate_retrieval ─┬─► ContextSufficient ─► synthesize [LLM] ─► AnswerDrafted ─► validate_answer ─► StopEvent
                                             └─► ContextInsufficient ─► plan_retry ─┬─► RephraseRequested ─► rephrase_query [LLM] ─► SearchRequested ─► retrieve
                                                                                     ├─► SearchRequested (expand top_k) ─► retrieve
                                                                                     └─► StopEvent ("לא נמצא מידע מספיק")
```
תרשים אינטראקטיבי: `docs/workflow_query.html`, `docs/workflow_extraction.html`, `docs/workflow_ingestion.html`.

## סכמת השכבה המובנית

### שדות משותפים לכל פריט
| שדה | הסבר |
|---|---|
| `id` | מזהה יציב: `hash(item_type + relative_path + topic + התוכן העיקרי המנורמל)`. בלי header ושורות, כך שאותו פריט שנמצא בשני סעיפים של אותו קובץ מתאחד |
| `item_type` | `decisions` / `rules` / `warnings` / `changes` |
| `topic` | מפתח נושא קצר (snake_case). מקבץ פריטים לצורך זיהוי עדכניות |
| `summary` | משפט אחד באנגלית. חובה: פריט בלי summary נפסל |
| `tags` | 2–5 מילות מפתח, מנורמלות ל-lowercase. משמשות לסינון |
| `area` | צוות או רכיב (`inbox`, `lens/frontend`, `relay/helm`...) |
| `source` | `tool`, `relative_path`, `header` (כותרת הסעיף), `line_range` (שורות 1-based בקובץ המקור). חובה |
| `observed_at` | תאריך הפריט: תאריך שכתוב בקובץ (`Date:`), אחרת תאריך השינוי של הקובץ |
| `date_source` | `content` או `file_mtime`: מאיפה הגיע התאריך |
| `status` | `active` או `superseded` |
| `supersedes` | רשימת ids של פריטים ישנים יותר באותו `(item_type, topic)` שהפריט הזה מחליף |

### שדות ייעודיים
| סוג | שדות | דוגמה מהקבצים |
|---|---|---|
| `decisions` | `decision`, `rationale` | topic אחד של Kafka לכל מקורות ה-bridge |
| `rules` | `rule`, `strength` (`must`/`should`) | "לא להעתיק את קבצי ה-proto לתיקיית הצוות" |
| `warnings` | `risk`, `kind` (`sensitive`/`limitation`/`do_not_touch`/`stale_info`) | ערך שבור של צוות אחד חוסם sync לכולם |
| `changes` | `subject`, `from_value`, `to_value` | `/api/auth/login` → `/auth/login` |

### מבנה הקובץ (`data/structured_store.json`)
```json
{
  "schema_version": "1.0",
  "generated_at": "...",
  "sources": [{"tool": "...", "root_path": "...", "files": [{"relative_path": "...", "last_modified": "...", "hash": "sha256"}]}],
  "items": {"decisions": [], "rules": [], "warnings": [], "changes": []}
}
```

### StructuredQuery (מה שה-LLM מחזיר במסלול המובנה)
`item_types`, `tags`, `tool`, `area`, `date_from`, `date_to`, `only_active`, `sort` (`newest`/`oldest`), `limit`.
זה אובייקט Pydantic, לא קוד ולא SQL. `validate_structured_query` בודק אותו לפני שהוא רץ.

## State לעומת Event
- **Event** מכיל רק את מה שה-Step הבא צריך. למשל `SearchRequested(query_text, top_k)` או `StructuredQueryValid(query, date_from, date_to)`.
- **State** מכיל את מה שנדרש לאורך כל הריצה. בשאילתה: `original_query`, `route`, `attempt`, `tried_queries`, `strategies_used`, `collected_nodes`, `structured_query`, `build_attempts`, `fallback_reason`, `stop_reason`. בחילוץ: `file_infos`, `expected_sections`, `skipped`, `duplicates_merged`.

## החלטות
1. **ה-Query נבנה ישירות כ-Workflow.** שלב ה-MVP של השאילתות (שאילתה, LLM ו-Gradio) לא היה קיים בקוד, ולכן לא היה מה "להמיר". נבנה מההתחלה לפי החלוקה שאושרה.
2. **LLM: Cohere `command-a-03-2025`.** המודל החדש ביותר בתיעוד של Cohere הוא `command-a-plus-05-2026`, אבל `llama-index-llms-cohere==0.10.0` מחזיק רשימת מודלים סגורה ודוחה אותו ("not supported for chat"). `command-a-03-2025` הוא החדש ביותר שהאינטגרציה הרשמית מקבלת, ו-Cohere מתארים אותו כמודל ל-RAG ולשימוש רב-לשוני.
3. **Embedding: `embed-v4.0`.** Cohere ממליצים היום על `embed-v5.0-*`, אבל האינטגרציה הרשמית של LlamaIndex (`llama-index-embeddings-cohere==0.10.0`) מקבלת רק עד v4.
4. **מודלים כפולים בחילוץ: Raw* מקלים מול פריטים קפדניים.** ה-LLM מחזיר מודלים שכל השדות בהם אופציונליים. כך פריט שבור אחד לא מפיל סעיף שלם. הקוד בונה מכל פריט מודל קפדני בנפרד, ופריט שנכשל בולידציה מדולג עם warning.
5. **source, line_range ותאריך נקבעים בקוד, לא ב-LLM.** ה-LLM לא יודע מספרי שורות, ועלול להמציא אותם. הם נלקחים מהפיצול לסעיפים.
6. **חילוץ לפי סעיף, עם fan-out.** `split_into_sections` שולח `SectionToExtract` לכל סעיף, ו-`validate_items` אוסף את כולם עם `collect_events`. סעיף שנכשל עדיין מחזיר `SectionExtracted` עם שגיאה, כדי שהאיסוף לא ייתקע. סעיפים של כותרת בלבד (פחות מ-`min_section_chars`) לא נשלחים ל-LLM.
7. **עדכניות לפי `(item_type, topic)`.** בכל קבוצה, הפריט עם ה-`observed_at` המאוחר ביותר הוא active, והישנים יותר superseded. החישוב נעשה מחדש על כל המאגר בכל ריצה.
8. **ריצה חוזרת:** קובץ שה-sha256 שלו לא השתנה לא נשלח ל-LLM. לקובץ שהשתנה, כל הפריטים שלו מוחלפים, כדי שלא יישארו פריטים ישנים.
9. **Repository אחד** הוא המקום היחיד שקורא וכותב את ה-JSON, כולל לוגיקת הסינון. מעבר ל-DB פירושו להחליף את המחלקה הזו בלבד.
10. **ראוטר: `LLMSingleSelector` של LlamaIndex.** התיאורים והדוגמאות נמצאים ב-`prompts/routes.yaml`, כך שאפשר לכוונן בלי לגעת בקוד. אם ה-Selector נכשל, ברירת המחדל היא המסלול הסמנטי, שתמיד עובד.
11. **השאילתה המובנית היא אובייקט, לא קוד או SQL.** כך אין הרצה של קוד שה-LLM כתב, והולידציה פשוטה ודטרמיניסטית: סוגים, כלי, תאריכים ו-limit. אם השאילתה לא תקינה, יש בנייה מחדש אחת עם השגיאות כמשוב, ואחריה fallback.
12. **Fallback כ-Event ייעודי (`FallbackRequested`) עם Step משלו.** כך הסיבה נרשמת ב-State במקום אחד, ומוצגת למשתמש.
13. **אותו כלל grounding בשני המסלולים:** התשובה מוצגת רק אם היא לא ריקה ומצטטת לפחות `[n]` אחד אמיתי. רשימת המקורות כוללת רק את מה שצוטט.
14. **התאריך של היום מוזרק ל-Workflow** (`today=`), כדי ששאלות כמו "בשבוע האחרון" ייבדקו באופן דטרמיניסטי.
15. **שדה `records` ולא `items` ב-Events.** `Event` הוא אובייקט דמוי dict, ושדה בשם `items` היה מסתיר את המתודה `.items()`.
16. **Structured extraction במצב JSON (`PydanticProgramMode.LLM`) ולא function calling.** ה-API הוא `llm.astructured_predict` של LlamaIndex. האינטגרציה של Cohere משטחת אובייקטים מקוננים בסכמת הכלי: רשימת אובייקטים חזרה כרשימת מחרוזות, וכל סעיף נכשל. במצב JSON, LlamaIndex שם את סכמת ה-Pydantic ב-prompt ומאמת את התשובה מולה.
17. **קריאות LLM דרך `achat` ולא `acomplete`.** ב-Cohere, `acomplete` קורא ל-endpoint הישן `/generate`, שמחזיר 404 למודלי Command A.
18. **ב-prompt החילוץ, שדות החובה של כל סוג נכתבים במפורש.** בריצה הראשונה ה-LLM מילא רק את השדות המשותפים, והולידציה פסלה 57 פריטים. הולידציה לא רוככה; ה-prompt תוקן.
19. **Few-shot בבניית השאילתה המובנית.** בלי דוגמאות, ה-LLM הוסיף tags כלליים ("technical") לשאלות "תן לי את כל...", והסינון החזיר 0 תוצאות.
20. **התאמת tags מנורמלת**, גם מול כותרת הסעיף ונתיב הקובץ. כך `code_review` תואם ל-`code-review.md`.
21. **`min_score: 0.20` מכויל על נתונים אמיתיים.** שאלות לא קשורות הגיעו לכל היותר ל-0.17, ושאלות רלוונטיות בעברית ל-0.21–0.62. הערך 0.35 שהוגדר מראש היה גורם לכל שאילתה סמנטית למצות ניסיונות.
22. **שפת התשובה נקבעת בקוד.** אם יש בשאלה אותיות עבריות, ה-prompt אומר במפורש "Answer in Hebrew". בלי זה, תשובה אחת יצאה באנגלית.
23. **Retry ל-429 בכל קריאת LLM** (6, 12 ו-24 שניות). מפתח trial של Cohere מגביל את קצב הבקשות. בלי retry, 429 בראוטר היה נבלע ומנתב בשקט לסמנטי.
24. **קובץ שאחד הסעיפים שלו נכשל נשמר בלי hash**, כך שהריצה הבאה מנסה אותו שוב ולא מסמנת אותו כ"לא השתנה".
25. **TLS דרך מאגר התעודות של Windows (`truststore`), לפי `network.use_system_cert_store`.** NetFree מיירט HTTPS, ו-Python 3.13 דוחה את ה-CA שלו במצב strict. התעודות עדיין נבדקות במלואן, על ידי Windows. זה גם מה ש-pip עושה.

## תוצאות (ריצה אמיתית, 2026-11-15)
- **Ingestion:** 3 קבצים → 46 Nodes → 46 וקטורים ב-Pinecone (`check_ingest.py`).
- **Extraction:** 37 סעיפים נשלחו ל-LLM (7 סעיפים של כותרת בלבד דולגו), ונשמרו **65 פריטים**:

  | סוג | claude_code | kiro | סה"כ |
  |---|---|---|---|
  | decisions | 8 | 2 | 10 |
  | rules | 14 | 15 | 29 |
  | warnings | 3 | 4 | 7 |
  | changes | 19 | 0 | 19 |
  | **סה"כ** | **44** | **21** | **65** |

  29 פריטים קיבלו `date_source=content` (התאריך `Date: October 20, 2026` בסיכום), ו-36 קיבלו `file_mtime`. ריצה חוזרת: 0 קריאות LLM, כל הקבצים "unchanged".
- **Router eval:** 8/8 (5 מובנות ו-3 סמנטיות). בהרצה המלאה (`--full`), 7 שאלות נענו עם מקורות. שאלת "בשבוע האחרון" עברה ל-fallback וקיבלה תשובה כנה "לא נמצא מידע", כי הקבצים מספטמבר–אוקטובר.
- **pytest:** 36/36, בלי רשת.

## מגבלות ידועות
- **סיווג רועש:** פריטים מהצ'קליסט של code-review סווגו לפעמים כ-decisions או כ-warnings במקום כ-rules. נוסף על כך נמצא change שגוי אחד: "MSW bypass rule: present → unchanged".
- **עדכניות עדיין לא נבחנה על נתונים אמיתיים.** כל ה-65 פריטים active, כי אין כרגע שתי גרסאות של אותו נושא בתאריכים שונים. הלוגיקה עצמה נבדקת ב-unit test. היא גם תלויה בכך שה-LLM ייתן אותו `topic` לאותו נושא, וזה לא מובטח.
- **0 כפילויות אוחדו בפועל**, כי ה-id כולל את התוכן המנורמל, וניסוחים שונים של אותו פריט מקבלים id שונה.
- **המרווח בין רלוונטי ללא רלוונטי בחיפוש הסמנטי צר** (0.17 מול 0.21). אם מוסיפים קבצים, צריך לכייל מחדש.
- **embed-v4.0 ו-command-a-03-2025** נבחרו כי אלה הגרסאות שהאינטגרציות הרשמיות תומכות בהן, לא כי הם החדשים ביותר אצל Cohere.
- מפתח trial: החילוץ המלא לוקח כ-2–3 דקות, כולל המתנות ל-429.

## דיון: עיצוב ה-Workflow

## דיון: מתי מבני ומתי סמנטי

