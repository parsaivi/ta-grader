# AGENTS.md

Guidance for coding agents (and people) changing this repository.

## What this is

`ta-grader` grades homework PDFs with two LLMs and sends disagreements to a
human review console. Package code is in `src/ta_grader/`; course data lives in
workspaces made by `ta-grader init`, never in this repository.

## Layout

| Path | Purpose |
|---|---|
| `cli.py` | The `ta-grader` command. Subcommands delegate to each module's own `main()` |
| `config.py` | Settings: environment, then `course.toml`, then defaults. Every module reads from here |
| `locales.py`, `data/locales/` | Language packs: heading words, digits, part labels, RTL |
| `hybrid_grader.py` | Default route: assess each PDF, send it to text or vision |
| `vision_grader.py` | Vision route: pages to two models; the only route with parts, quotes, posture, samples |
| `pipeline.py`, `homework_grader.py`, `question_splitter.py` | Text route: extract, split, grade |
| `smart_pdf_extractor.py`, `pdf_extractor.py` | Text extraction; vision only for figures |
| `grade_parsing.py` | Reading grades, tags and evidence out of model output; conflict resolution |
| `assurance.py` | Vetoes (illegible, not found), sampling confidence, evidence checks |
| `posture.py`, `data/postures/` | Marginal-call rules, from a closed vocabulary |
| `tagging.py`, `data/tagsets/` | Labels recorded next to grades, never scored |
| `review_store.py` | Review data layer: queue, decisions, ledger, sign-off, export |
| `review_app.py`, `review_entry.py` | The Streamlit console |
| `benchmark_models.py` | Cost and grades per model on real PDFs |
| `demo.py` | The synthetic demo course |

## Commands

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                    # offline
.venv/bin/ruff check src tests
ta-grader demo /tmp/demo && cd /tmp/demo && ta-grader route -q 1
```

## Rules

Read `docs/design.md` before changing grading or review code. The short
version:

- `null` and `0` must stay distinct everywhere, exports included. Test for a
  missing grade with `is not None`, never truthiness.
- Grades come only from an explicit `GRADE:`/`GRADES:` line.
- Both graders must run unless `--single` is given, and single runs say so.
- Postures and tag schemes cannot contribute free prompt text. Grade-changing
  rules belong in the marking scheme.
- Tags never affect grades.
- Results are written atomically; result dicts stay uniformly shaped,
  including on error paths.
- Nothing exports without a sign-off.
- Nothing language-specific goes in code; it goes in a locale pack.
- Nothing course-specific goes in code; it goes in a rubric, posture or
  tag scheme.
- Never add student data, real ids or API keys to the repository, tests or
  fixtures. Use `ta-grader demo` or invented data.

`config.py` values are captured by `from .config import X` at import time. The
CLI resolves the workspace and reloads config before importing anything else;
keep it that way when adding subcommands.

The review console's layout fights Streamlit in specific ways; read
`docs/review-console.md` before touching its CSS.
