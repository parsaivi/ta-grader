# Homework Grading System

## Project Overview

AI-powered homework grading for a Persian/English CE course. Two LLMs grade
every question independently; anything they disagree on is flagged for human
review rather than resolved silently.

Three entry points:
- **Hybrid** (`hybrid_grader.py`) — assesses each PDF and routes it to text or
  vision. **This is the default path.**
- **Vision** (`vision_grader.py`) — PDF pages straight to vision models. Reads
  anything, costs the most.
- **Text** (`pipeline.py`) — extract → split → grade. Cheapest, but grades
  handwritten submissions as blank, so never run it alone across a whole class.

## Directory Structure

```
TA project/
├── students_pdf/              # INPUT: student PDFs (filename stem = student_id)
├── answers/                   # INPUT: correct_answer_N.md marking schemes
├── tagsets/                   # INPUT: optional label vocabularies (JSON)
├── postures/                  # INPUT: marginal-call rule sets (JSON)
├── students_extracted/        # AUTO: full text extractions (text route)
├── questions/question_N/      # AUTO: per-question answers (text route)
├── .env                       # LLM_KEY=...
└── grades_question_N.json     # OUTPUT
```

Note the marking-scheme folder is `answers/`. An earlier version of
`pipeline.py` looked for `correct_answers/`, which does not exist and made the
grade step fail unconditionally.

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings; every module imports from here |
| `hybrid_grader.py` | Routes each student to text or vision, merges results |
| `benchmark_models.py` | Measures grade/cost/latency per model on real PDFs |
| `grade_parsing.py` | Grade + tag extraction from model output, conflict resolution |
| `tagging.py` | Tag schemes: loads `tagsets/*.json`, renders their prompt block |
| `posture.py` | Grading posture: loads `postures/*.json`, renders its prompt block |
| `assurance.py` | Legibility/presence vetoes, sampling confidence, evidence checks |
| `vision_grader.py` | Primary grader (PDF pages → two vision models) |
| `pipeline.py` | Text-route orchestrator |
| `smart_pdf_extractor.py` | Hybrid extraction: text direct, vision only for figures |
| `pdf_extractor.py` | Full vision extraction (expensive) |
| `question_splitter.py` | Locates question boundaries |
| `homework_grader.py` | Text-based dual-model grader |
| `grading_prompts.py` | Grading system prompt |
| `review_app.py` | Review console (Streamlit): triage, cohort, fairness, ledger, sign-off |
| `.streamlit/config.toml` | The console's pinned light theme — see the review console section |
| `review_store.py` | Review data layer: items, queue order, decisions, ledger, export |
| `conflict_reviewer.py` | Older terminal review UI (single-question files only) |
| `test_grading.py` | Tests (no API key needed) |

## Configuration

All configuration lives in `config.py`, overridable via `.env`. Do not
reintroduce per-module `API_KEY`/`API_URL`/model constants — they were
duplicated across five files and drifted.

```bash
LLM_KEY=sk-or-v1-...            # required, OpenRouter
LLM_URL=https://openrouter.ai/api/v1
MODEL_1=openai/gpt-4.1-nano
MODEL_2=google/gemini-2.5-flash
SPLITTER_MODEL=google/gemini-2.5-flash
VISION_MODEL=google/gemini-2.5-flash
GRADE_CONFLICT_LIMIT=10
GRADE_RESOLUTION=max            # max | mean | min
QUESTION_PARTS=                 # e.g. a,b,c -- empty = grade question whole
TAG_SCHEME=                     # e.g. sql_constructs -- empty = no tagging
POSTURE=                        # e.g. strict -- empty = built-in defaults
GRADE_SAMPLES=1                 # >1 grades each submission N times for confidence
MAX_WORKERS=5
MIN_CHARS_PER_PAGE=400          # below this -> vision route
MIN_ANSWERED_FRACTION=0.25      # splitter yield below this -> vision route
```

Requests go through OpenRouter, so model ids are provider-prefixed. Both
grading models must accept image input. See `.env.example`.

## Commands

```bash
# Hybrid (default) -- routes per student
python hybrid_grader.py --question 8 --num-questions 10
python hybrid_grader.py --question 8 --dry-run     # preview routing, free
python pipeline.py --step hybrid --question 8 --num-questions 10

# Compare model cost/quality on real submissions before committing
python benchmark_models.py --question 8 --limit 3

# Vision route
python vision_grader.py --question 8
python vision_grader.py --num-questions 10
python vision_grader.py -q 8 --conflict-limit 5 --resolution mean --dpi 200

# One question marked part by part, each part graded 0-100 independently
python vision_grader.py -q 2 --parts a,b,c --pdf-folder students_pdf_hw3

# ...with a course-specific label recorded next to each grade (never affects it)
python vision_grader.py -q 2 --parts a,b,c --tags sql_constructs

# ...with the marginal-call rules stated explicitly (these DO affect grades)
python vision_grader.py -q 2 --parts a,b,c --posture strict

# ...graded 3x to measure how much the model agrees with itself
python vision_grader.py -q 2 --parts a,b,c --samples 3

# Single grader (half the cost, no cross-check) and a bounded sample
python vision_grader.py -q 2 --parts a,b,c --single \
    --model-1 google/gemini-2.5-flash --limit 20

# Text route
python pipeline.py --step extract --pdf-folder students_pdf --smart-extract
python pipeline.py --step split --num-questions 10
python pipeline.py --step grade --question 8
python pipeline.py --step all --num-questions 10 --question 8 --smart-extract

# Splitter inspection (cheap)
python question_splitter.py --preview students_extracted/400000003.txt -n 10
python question_splitter.py -n 10 --no-llm      # zero API calls

# Review
streamlit run review_app.py
python conflict_reviewer.py --question 8   # older, single-question files only

# Tests
python test_grading.py
```

## Part-by-part grading

`--parts a,b,c` grades one question's parts separately. Each part gets its own
0-100 grade, its own conflict flag and its own column in the summary; results
carry both `final_grades` (ordered list) and `part_grades` (keyed by label).
Two models can agree on part (a) and disagree sharply on part (c), and
collapsing that into one question-level flag hides the disagreement a human
needs to see.

This is not the same as `--num-questions 3`: the parts share one statement and
one marking scheme, and a student's answers to all of them usually sit in one
block of the page, so the model sees the whole submission and then splits its
judgement.

Parts are opt-in. Without `--parts` (or `QUESTION_PARTS`) the question is
graded as a whole and no part machinery runs.

## Tag schemes

A **tag** is a course-specific observation recorded next to a grade -- "this
query uses a CTE", "this proof is by induction". The vocabulary lives in a JSON
file under `tagsets/`, not in the prompt builder, so a new course adds a file
instead of editing `vision_grader.py`.

```bash
python vision_grader.py -q 2 --parts a,b,c --tags sql_constructs
python vision_grader.py -q 4 --parts a,b   --tags ./my_scheme.json
python vision_grader.py -q 2 --parts a,b,c            # no tagging at all
```

A scheme defines `key` (results key), `title`, `field` (the output line the
model writes, e.g. `CONSTRUCTS`), `labels` with a one-line `description` and a
one-char `marker` for the summary grid, optional `aliases` for variants the
model tends to invent, and optional `notes` appended to the prompt section.
Shipped: `sql_constructs`, `proof_technique`.

Results carry `tag_scheme` (which vocabulary) and `part_tags` (the labels).
Without `--tags` neither key is written and the prompt never mentions labels --
an empty scheme is a real state, not a scheme with zero labels.

Two properties are deliberately **not** configurable:

- **Tags never affect grades.** `TagScheme.prompt_section()` always appends the
  "NOT penalised, must NOT lower the grade" boilerplate, and a scheme file
  cannot suppress it. A model told to notice a construct will otherwise quietly
  start deducting for it; this project has already had that bug. If a course
  genuinely wants to penalise something, that belongs in the marking scheme in
  `answers/`, where it is visible as a grading criterion.
- **Unknown labels are discarded.** Only the scheme's vocabulary is stored, so
  downstream filters can treat it as exhaustive.

## Grading posture

A **posture** decides the marginal calls the marking scheme does not settle:
which band to award when an answer falls between two, whether a materially
correct answer that differs from the reference gets full marks, whether a syntax
slip costs anything, which of two attempts counts.

```bash
python vision_grader.py -q 2 --parts a,b,c --posture strict
python vision_grader.py -q 2 --parts a,b,c --posture ./my_posture.json
python vision_grader.py -q 2 --parts a,b,c              # built-in defaults
```

Shipped: `lenient`, `standard`, `strict`. Parameters are `boundary_rule`,
`near_miss`, `approach_vs_result`, `minor_errors`, `multiple_attempts`,
`ambiguous`, `presentation`, `language`, plus `unattempted` (whether a blank
part records 0 or null).

Four properties are deliberate:

- **A posture is never absent**, only unstated. `--posture` omitted loads
  conservative defaults rather than nothing, because a prompt silent on how to
  treat a borderline answer does not get neutrality — it gets whatever the
  model felt like that student. `default_posture()` is the floor.
- **Every parameter is always stated in the prompt, defaults included.** Do not
  "optimise" this by emitting only non-default choices. Silence about
  presentation or language is how a model starts quietly deducting for messy
  handwriting or for answering in Persian; the `ignore` values are emphatic
  prose, not omissions.
- **Postures select from a closed vocabulary and cannot contribute prompt
  text.** The sentence for each value lives in `posture.py`. A posture file that
  could write free text would be a channel for course-specific grading rules,
  and those belong in `answers/correct_answer_N.md` where a human reviews them
  before the run. Unknown parameters and unknown values are hard errors.
- **The full set of choices is recorded in the results**, not just the name.
  Storing `"strict"` alone would silently misdescribe a run's marks after any
  later edit to `strict.json`.

Posture changes grades, so it must not change mid-round: a cohort graded under
two postures is unfair by construction. Changing it means regrading everyone.

## Assurance: what a grade is worth

`assurance.py` holds three mechanisms that decide how much to trust a grade.
Every one of them can withhold or flag a grade and none can raise one — a signal
that could raise a mark would be a grading criterion, and those live in
`answers/`.

**Legibility and presence can veto a grade.** The model reports, per part,
whether it could read the answer (`clear`/`partial`/`illegible`) and whether the
answer is there at all (`answered`/`unattempted`/`not_found`). Then:

| Reported | Recorded | Why |
|---|---|---|
| `not_found` | `null` + flag | The part is not in the pages. A page that failed to scan is not a wrong answer. |
| `illegible` | `null` + flag | Cheap models return 0 for Persian handwriting they cannot read; this makes the admission mandatory and visible. |
| `unattempted` | `0` | The answer's place was found and was blank, so 0 is a claim we can actually make. Posture `unattempted: null` preserves the distinction instead. |

This is ADR-003 (`null` ≠ `0`) pushed back to its cause. Vetoes are applied
*after* resolution and override both models: two models agreeing confidently on
a grade for a page that is not in the file are both wrong.

**Sampling measures confidence.** `--samples N` grades each submission N times
with model 1; the spread is the confidence, thresholded at `conflict_limit`
(`high` ≤ limit, `medium` ≤ 2×, else `low`). The final grade is the **median**,
not the max — the max of N noisy draws is a bias that grows with N, not an
estimate. Do not ask the model for a self-reported confidence number instead;
those cluster in the high eighties regardless of correctness, and an
uncalibrated confidence is worse than none because it gets trusted. Confidence
is not reported for vetoed parts.

**Evidence quotes detect confabulation.** Each part carries a verbatim quote of
the line the grade turns on. A quote that is missing, or that turns out to
appear in the marking scheme rather than the submission, is flagged: that is
exactly what `gpt-4.1-nano` did when it asserted "no subqueries or CTEs" about
pages beginning `WITH E AS (...)`. Catching that once took a human rendering
page 4 to PNG; `check_evidence()` now does it on every grade.

All three are wired into the **per-part path only**, like tagging. `--samples`
without `--parts` is an argument-parse error rather than a silent no-op.

An unreported label is `None`, distinct from a label named `"none"`: the first
means the model did not say, the second means it looked and found nothing.

`_result_tags()` also reads the older `part_constructs` key, so results written
before schemes were configurable still print correctly on a resumed run.

## Single-model mode

`--single` runs `--model-1` only: half the cost, no cross-checking. It is an
explicit flag and is recorded as `"single_model": true` in the output, with
`model_2.name` set to null. Do not achieve the same effect by pointing both
models at the same id -- results would then claim dual-model verification that
did not happen. In single mode the only thing flagged is a part the model
failed to grade at all.

## Cost reporting

Every call's tokens and cost are accumulated in a `UsageLedger` and written to
the output JSON under `usage`, plus printed as a table with a per-student
figure and a per-class extrapolation. Cost comes from the gateway's own
accounting (`extra_body={"usage": {"include": True}}`), not a local price
table: image tokenization varies ~6x between model families, so a
price-per-token estimate is not usable. A gateway that reports no cost leaves
the dollar figures at 0; the token counts are still real.

## Grading Logic

1. Both models grade independently, 0-100.
2. Difference ≤ `GRADE_CONFLICT_LIMIT` → resolved per `GRADE_RESOLUTION`.
3. Difference > limit → flagged (`conflict: true`).
4. Exactly one model produced a grade → flagged. The surviving grade is kept
   but is not an agreed result.
5. Neither model produced a grade → `null` and flagged.

`null` and `0` mean different things and must stay distinguishable: `0` is
"the student got it wrong", `null` is "we do not have a grade". Never
substitute one for the other.

## Invariants worth preserving

These encode bugs that were fixed; the tests in `test_grading.py` cover each.

- **Both graders must actually run.** A previous version had model 1 commented
  out and `conflicts` hardcoded to `False`, which made every grade single-model
  and left the conflict reviewer with nothing to review.
- **Never infer a grade from prose.** Grade extraction matches an explicit
  `GRADE:` / `GRADES:` line only. The old fallback took "the last number
  between 0 and 100 anywhere in the response", which turned phrases like
  "6 transistors" into grades.
- **The splitter must not echo the homework.** It asks the model for start
  *line numbers* and slices locally. The old design had the model reproduce
  every answer as JSON within `max_tokens=4096`; long submissions truncated,
  parsing failed, and the handler quietly returned empty answers for every
  question — recording real work as unanswered.
- **Silent failure is not acceptable.** The splitter reports per-student counts
  and warns when a submission yields nothing.
- **Result dicts are uniformly shaped.** Error paths must include `model_1` and
  `model_2`; the summary printers and Streamlit app index them unconditionally.
- **0 is a real grade.** Do not use truthiness (`if g1 and g2`) to test for a
  missing grade — use `is not None`.
- **Persian RTL word order.** Extractors reverse predominantly-RTL lines to
  restore logical order; the splitter recognises both `سوال ۱` and the
  extraction artefact `۱ سوال`.
- **Vision payloads must stay under the API image limit.** Pages render as
  JPEG and the DPI steps down until the total fits `MAX_IMAGE_MB`. A 17-page
  scan at 150 DPI PNG was 31 MB and got a hard 413, failing the exact students
  the vision route exists to serve.
- **A successful call can still be truncated.** Four HW3 submissions returned
  4-197 characters and stopped mid-sentence. No exception was raised and
  `failed` stayed 0, so it looked like a model that declined to grade rather
  than a cut-off stream. `call_vision_llm` now prints `finish_reason` whenever
  it is not `stop`. Dropping the ungraded rows from the results file and
  re-running regrades exactly those students, since runs resume by student id.

- **Course-specific rules belong in data, not in the prompt builder.** The HW3
  run needed "note whether the student used a CTE", and the first version of it
  hardcoded `cte`/`subquery` into the grade parser, the prompt and the summary
  markers -- three files to edit for the next course. Rules that change the
  grade go in `answers/correct_answer_N.md`; observations recorded next to the
  grade go in a `tagsets/*.json` scheme. `vision_grader.py` should not name a
  construct from any particular course.
- **Handwritten submissions must not reach the text grader.** They extract to
  nothing and would be scored as blank -- a zero for a student who did the
  work. `hybrid_grader` checks chars/page before grading and splitter yield
  after, routing failures to vision. Do not bypass it for a whole-class run.
- **A grade of 0 is a claim about the student's work, and may only be recorded
  when there is work to make a claim about.** An answer the model could not read
  (`illegible`) or could not find in the file (`not_found`) records `null` and
  flags, never 0. Both are the same injustice as the text-route bug above,
  reached by different routes: a mark of zero against work that exists on a page
  that did not scan, or in handwriting the model could not parse. `apply_vetoes`
  enforces this after resolution, overriding both models.
- **Confidence must be measured, never self-reported.** Repeated sampling of one
  model, with the spread as the signal. A model asked "how confident are you"
  answers in the high eighties whether or not it is right, and a confidence
  number that is trusted and wrong is worse than none at all. The motivating
  case is real: one HW3 submission scored 40 and then 20 on a re-run of the same
  model at temperature 0 — twice the limit used to decide two *different* models
  disagree.
- **Repeated samples combine by median, not by max.** `GRADE_RESOLUTION=max` is
  defensible for two agreeing models and wrong across N draws from one: taking
  the maximum is an upward bias that grows with N.
- **Grading posture is stated in full, always, and recorded with the results.**
  Including the defaults; see the posture section above for why silence is not
  neutrality. A posture must not change mid-round.

### The review console can break all of these too

`review_app.py` writes the same results file the grader writes, so every rule
above applies to it. The ones the version it replaces had already broken:

- **Accepting "the higher grade" where there is no grade must yield `null`.**
  The old `pick_higher` ended `... if present else 0`, so resolving a part
  neither model could grade wrote a zero against work nobody had looked at —
  ADR-003 violated through the review tool rather than through the grader.
- **The reviewer writes atomically and checks the file's mtime first.** A
  grading run may still be appending to it; the old reviewer held the whole file
  in session state and rewrote it on every save, which would delete any student
  a resumed run had added.
- **An override never overwrites the model's judgement, it records beside it.**
  Decisions append to `grades_*.review.jsonl` with the original grade, the new
  one, a reason and a timestamp. This is the only artifact here with value past
  the round: a growing set of human-corrected grades, which is the ground truth
  needed to measure accuracy at all.
- **A withheld grade exports as an empty cell, never as 0.** A spreadsheet is
  exactly where the null-vs-zero distinction gets destroyed.
- **Export is gated on batch sign-off.** No mark reaches a student without a
  human accepting the batch.

### Working on the console's layout

The console is a Streamlit app wearing a designed layout, and Streamlit fights
that in four specific ways. All four have already cost a day; read them before
touching `CSS` in `review_app.py`.

- **Emotion class selectors out-specify yours.** Streamlit styles blocks with
  two chained classes, so a plain `.st-key-foo {height: 40px}` loses. Layout
  declarations need `!important`.
- **`height` loses to `flex`.** Every block is `flex: 1 1 0%` inside a flex
  column, so a block in a parent with a height *grows to fill it* whatever
  height you set. A bar that must stay 40px needs `flex: 0 0 auto !important`
  on the block **and** on the `stLayoutWrapper` that Streamlit puts around it.
- **`stColumn:first-child` is a descendant selector.** Written without `>` it
  also matches the first column of every row nested inside that column, which
  is how the 40px page bar once inherited `height: calc(100vh - 110px)`. Scope
  chrome rules with `>` the whole way down.
- **Nested `st.columns` inherit the parent's height and wrap.** A row inside a
  row in a fixed-height bar lays out as a staircase of half-height columns.
  Keep bars flat: one row, one widget per column.

Two more that are not CSS:

- **`label {display: none}` hides toggles.** `st.toggle` renders its control
  *inside* the `<label>`. Hide labels per widget type, or pass
  `label_visibility="collapsed"`.
- **Keyboard shortcuts must bind in both documents.** The listener is injected
  by `components.html`, which is an iframe; a key press goes to whichever frame
  the browser has focused, and on a fresh load that is the iframe. `KEYS_JS`
  binds `keydown` on the parent document *and* on its own, and ignores
  `!e.isTrusted` so it never reacts to an event it dispatched itself.

Typed digits go through hidden buttons (`digit_0`..`digit_9`), not into the
number field. A value typed into a Streamlit widget is uncommitted until blur,
so anything that acts on it immediately reads the *previous* number. A button
click is the one interaction Streamlit cannot lose.

## Cost

Vision cost is dominated by how a model tokenizes page images, and that varies
by an order of magnitude between families -- it is not predictable from the
per-token price. Measured on one 9-page submission:

| Model | Prompt tokens | Cost |
|-------|---------------|------|
| `openai/gpt-5.6-luna` | 25,089 | $0.0336 |
| `google/gemini-2.5-flash` | 4,276 | $0.0020 |

Same pages, 6x the tokens and 17x the cost. Always measure with
`benchmark_models.py` before switching graders.

A model can also fail in a way that costs *more* and grades worse.
`openai/gpt-4.1-nano` on the HW3 SQL question spent 4.5x gemini's prompt
tokens, produced no parseable grade at all for 3 of 20 students, and on the
rest asserted "no subqueries or CTEs, complies with the restrictions" for
submissions whose pages plainly begin `WITH E AS (...)`. It was not reading the
pages; it was paraphrasing the reference solution back. Because
`GRADE_RESOLUTION=max`, its invented high grades then won the merge. When a
marking scheme turns on a construct being present or absent, check that the
model quotes the construct it claims to have found -- generic assurances of
compliance are the tell.

Cheaper is not automatically better: gemini-2.5-flash-lite, nova-lite and
mistral-small all returned 0 for Persian submissions they could not read, which
is indistinguishable in the output from a genuine zero. A candidate model must
produce varied, defensible grades on submissions you have marked yourself
before it goes anywhere near a real class.

## Dependencies

```bash
pip install -r requirements.txt
```

## Notes

- The venv must be recreated if the project directory moves — console-script
  shebangs embed the absolute path.
- Results are saved atomically (`.tmp` + `os.replace`) after each student, so
  runs resume cleanly and an interrupt cannot truncate the JSON.
