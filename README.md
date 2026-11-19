# ta-grader

Grade homework PDFs with two LLMs and review only the answers they disagree on.

Both models grade every answer against your marking scheme. Where they agree,
the grade is filled in. Where they disagree, can't read the page, or can't
quote the student's work to back up a grade, the answer goes to a review queue
for a TA. Nothing reaches a student until a TA signs off the batch.

It reads typed and handwritten submissions in any language you add a
[locale pack](docs/adding-a-language.md) for. It was built for Persian/English
computer engineering homework, where a class hands in a mix of typed Persian,
handwriting and hand-drawn diagrams, and has graded real rounds of 81 and 119
students.

## Try it in two minutes

```bash
pip install "ta-grader[review] @ git+https://github.com/parsaivi/ta-grader"
ta-grader demo            # a synthetic course: 7 invented students
cd ta-grader-demo
ta-grader route -q 1      # free: which PDFs can be read as text, which need vision
```

```
Routing 7 submissions:
  text route   : 6
  vision route : 1

Going to the vision grader:
  S1007        only 0 chars/page -- handwritten or scanned
```

To grade the demo for real, put an [OpenRouter](https://openrouter.ai) key (or
any OpenAI-compatible endpoint's key) in `.env` and run:

```bash
ta-grader grade --route vision -q 1   # a few cents for the class
ta-grader review                      # opens the review console
```

## Grading your own course

```bash
ta-grader init algorithms-fall
cd algorithms-fall
```

This makes a workspace:

```
algorithms-fall/
├── course.toml     settings: languages, models, posture, points per question
├── rubrics/q1.md   one marking scheme per question
├── submissions/    one PDF per student; the filename is the student id
├── results/        grades and the review ledger
└── .env            LLM_KEY=...  (never shared, never committed)
```

`course.toml` is meant to be shared between the TAs of a course. The API key
is never read from it. Everything holding student data (`submissions/`,
`results/`, `.env`) is in the workspace's `.gitignore`, and the workspace is
separate from this code, so student work cannot end up in a commit to this
repository.

A marking scheme is a Markdown file. The `\question` and `\answer` markers are
optional:

```markdown
\question
What is the worst-case running time of selection sort? Explain.

\answer
O(n^2): (n-1) + (n-2) + ... + 1 = n(n-1)/2 comparisons.
- 50: states O(n^2)
- 50: justifies it with the sum, or an equivalent argument
```

The more explicit the mark allocation, the less the two models disagree.

Then:

```bash
ta-grader route -q 1                     # free preview of the routing
ta-grader grade -q 1                     # hybrid route (cheapest)
ta-grader grade --route vision -q 1 --parts a,b --posture strict --samples 3
ta-grader review                         # resolve what was flagged, then sign off
ta-grader export results/grades_question_1_parts.json   # CSV, only after sign-off
```

Every option is in [docs/configuration.md](docs/configuration.md).

## How a grade is made

**Two models, one conflict limit.** Both models grade each answer from 0 to
100. If their grades are within `conflict_limit` (default 10), the grade is
resolved automatically (`resolution`: max, mean or min). If they are further
apart, or one model returned nothing, the answer is flagged. If both failed,
the grade is left `null` and flagged.

**A missing grade is not a zero.** `0` means the student got it wrong. `null`
means there is no grade. A page that did not scan, an answer the model could
not read (`illegible`), or a part it could not find (`not_found`) records
`null` and goes to review. Exports write it as an empty cell.

**Every grade quotes the work.** On the vision route each part carries a
verbatim quote of the line its grade turns on. A missing quote, or one that
comes from the marking scheme instead of the submission, is flagged. That is
how a model reciting the answer key gets caught. One model once said a student
used no CTEs when the whole answer was built on one.

**Confidence is measured.** `--samples 3` grades each submission three times
and reports the spread. The final grade is the median. Models are not
deterministic at temperature 0: one submission scored 40, then 20, on a re-run
of the same model.

**The marginal calls are stated.** A posture (`lenient`, `standard`,
`strict`, or your own file) decides what the marking scheme leaves open:
which band a borderline answer gets, whether a nearly-right answer gets full
marks, whether a syntax slip costs anything. It is written in full into every
prompt and recorded with the results.

The reasons behind each of these rules are in [docs/design.md](docs/design.md).

## Three routes

| | `--route hybrid` (default) | `--route vision` | `--route text` |
|---|---|---|---|
| How | Routes each student to text or vision | PDF pages to vision models | Extract, split, grade text |
| Handwriting | Yes, automatically | Yes | No: scored as blank |
| Parts, quotes, posture, samples | No | **Yes** | No |
| Cost | Lowest that is safe | Highest | Lowest |

The hybrid route checks each PDF first. A PDF with fewer than 400 extracted
characters per page, or one the splitter could not divide into answers, goes
to vision. On a real class of 119, 115 went the text route and 4 went to
vision, so nobody's handwriting was graded as blank.

Per-part grading, evidence quotes, postures and sampling run on the vision
route only. Use it when those matter more than the cost.

## The review console

`ta-grader review` opens a queue ordered by how likely a grade is to be wrong
times how many marks it is worth, so the riskiest grade comes first. It shows
the student's page next to the quote the grade turns on, and can be driven
from the keyboard. A TA can mark an answer blind before seeing the model's
grade. Every decision is appended to a ledger beside the model's original
grade. A Fairness view lists pairs of near-identical answers that got
different marks. See [docs/review-console.md](docs/review-console.md).

## Models and cost

Requests go to any OpenAI-compatible endpoint (`[api] url` in `course.toml`):
OpenRouter, OpenAI, vLLM, Ollama, or a university gateway. Both grading models
must accept images. Use two *different* models: two runs of one model agreeing
tells you little, and the tool warns if you try.

The defaults were picked by measurement on Persian/English submissions,
question 8, both graders:

| Model | $/student | Grades given | Verdict |
|-------|-----------|--------------|---------|
| `openai/gpt-4.1-nano` | 0.0015 | 85, 88 | default |
| `google/gemini-2.5-flash` | 0.0018 | 100, 75 | default |
| `openai/gpt-5.6-luna` | 0.0202 | 95, **0** | 13x the cost, graded no better |
| `qwen/qwen3-vl-32b-instruct` | 0.0016 | 100, **0** | unreliable on Persian |
| `google/gemini-2.5-flash-lite` | 0.0004 | **0, 0** | could not read Persian |
| `amazon/nova-lite-v1` | 0.0007 | **0, 0** | could not read Persian |
| `mistralai/mistral-small-3.2-24b` | 0.0014 | **0, 0** | could not read Persian |

The cheapest models return `0` when they cannot read a page, which looks
exactly like a student who got it wrong. Vision cost depends on how a model
turns page images into tokens, which varies about 6x between model families
and is not visible in the per-token price. Measure on your own submissions
before you switch:

```bash
ta-grader benchmark -q 1 --limit 3
```

A class of 119, all 10 questions, both graders, cost about $4 with the
defaults. Grading all questions in one pass (`--num-questions 10`) is much
cheaper than ten single-question runs, which send every page ten times.

## Languages

Question-heading words, native digits, part labels and right-to-left repair
come from locale packs in `src/ta_grader/data/locales/`. English and Persian
ship with the package. Enable them per course:

```toml
[course]
languages = ["fa", "en"]
```

Adding a language is one small TOML file: see
[docs/adding-a-language.md](docs/adding-a-language.md).

## Responsible use

This is an assistant for a TA, not a replacement for one. Read
[RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) before using it on a real class,
especially the part about sending student work to a model provider.

## Development

```bash
git clone https://github.com/parsaivi/ta-grader && cd ta-grader
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # offline, no API key
.venv/bin/ruff check src tests
```

The end-to-end test runs a whole round on the demo course with a stand-in
model client. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE).
