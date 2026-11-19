# Design notes

Why the grader behaves the way it does. Most of these rules exist because the opposite behaviour shipped once and failed real students. The tests in `tests/` cover each invariant.

## Decision records cited in the code

Comments in the source refer to these by number.

| ADR | Decision |
|---|---|
| ADR-001 | Every grade comes from two independent models. Both must actually run; their agreement is the check. |
| ADR-003 | `null` is not `0`. `0` says the student got it wrong; `null` says there is no grade. No code path may turn one into the other. |
| ADR-005 | A single model's grade is never presented as an agreed result. Single-model runs are explicit (`--single`) and recorded. |
| ADR-006 | Rules that change a grade live in the marking scheme, where a human reads them before the run. Settings files (postures, tag schemes) choose from closed vocabularies and cannot add prompt text. |
| ADR-010 | Results files are written atomically (temp file, then `os.replace`), so an interrupted run or a concurrent reviewer cannot truncate them. |

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
file under `tagsets/` (shipped, or the workspace's own), not in the prompt builder, so a new course adds a file
instead of editing `vision_grader.py`.

```bash
ta-grader grade --route vision -q 2 --parts a,b,c --tags sql_constructs
ta-grader grade --route vision -q 4 --parts a,b   --tags ./my_scheme.json
ta-grader grade --route vision -q 2 --parts a,b,c            # no tagging at all
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
  the rubric files, where it is visible as a grading criterion.
- **Unknown labels are discarded.** Only the scheme's vocabulary is stored, so
  downstream filters can treat it as exhaustive.

## Grading posture

A **posture** decides the marginal calls the marking scheme does not settle:
which band to award when an answer falls between two, whether a materially
correct answer that differs from the reference gets full marks, whether a syntax
slip costs anything, which of two attempts counts.

```bash
ta-grader grade --route vision -q 2 --parts a,b,c --posture strict
ta-grader grade --route vision -q 2 --parts a,b,c --posture ./my_posture.json
ta-grader grade --route vision -q 2 --parts a,b,c              # built-in defaults
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
  and those belong in the rubric file (`rubrics/q<N>.md`) where a human reviews them
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
the rubric files.

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

## Grading logic

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

These encode bugs that were fixed; the tests in `tests/` cover each.

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
  grade go in the rubric file (`rubrics/q<N>.md`); observations recorded next to the
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

## Cost

Vision cost is dominated by how a model tokenizes page images, and that varies
by an order of magnitude between families -- it is not predictable from the
per-token price. Measured on one 9-page submission:

| Model | Prompt tokens | Cost |
|-------|---------------|------|
| `openai/gpt-5.6-luna` | 25,089 | $0.0336 |
| `google/gemini-2.5-flash` | 4,276 | $0.0020 |

Same pages, 6x the tokens and 17x the cost. Always measure with
`ta-grader benchmark` before switching graders.

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

