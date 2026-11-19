# Configuration

Settings come from three places. Higher in the list wins:

1. Environment variables, usually from the workspace's `.env`.
2. `course.toml` in the workspace (the nearest folder at or above where you
   run `ta-grader` that holds one).
3. Built-in defaults.

Without a `course.toml`, everything resolves against the current folder, using
the older layout (`students_pdf/`, `answers/correct_answer_N.md`).

The API key is read only from the environment, never from `course.toml`, so
the course file can be shared.

## course.toml

| Key | Env variable | Default | Meaning |
|---|---|---|---|
| `[course] name` | | folder name | For your own reference |
| `[course] languages` | `LANGUAGES` | `["en"]` | Locale packs to enable |
| `[api] url` | `LLM_URL` | OpenRouter | Any OpenAI-compatible endpoint |
| | `LLM_KEY` | *(required)* | API key |
| `[models] grader_1` | `MODEL_1` | `openai/gpt-4.1-nano` | First grader |
| `[models] grader_2` | `MODEL_2` | `google/gemini-2.5-flash` | Second grader |
| `[models] vision_grader_1/2` | `VISION_GRADER_1/2` | the graders | Graders for the vision route when hybrid routes there |
| `[models] splitter` | `SPLITTER_MODEL` | `google/gemini-2.5-flash` | Finds where each answer starts (text route) |
| `[models] figures` | `VISION_MODEL` | `google/gemini-2.5-flash` | Describes figures during text extraction |
| `[grading] conflict_limit` | `GRADE_CONFLICT_LIMIT` | `10` | Grade gap (0-100) above which an answer is flagged |
| `[grading] resolution` | `GRADE_RESOLUTION` | `max` | `max`, `mean` or `min` when the models agree |
| `[grading] posture` | `POSTURE` | built-in defaults | `lenient`, `standard`, `strict`, or a path |
| `[grading] tags` | `TAG_SCHEME` | none | A label vocabulary recorded next to grades, never scored |
| `[grading] parts` | `QUESTION_PARTS` | none | Default part labels, e.g. `["a", "b"]` |
| `[grading] samples` | `GRADE_SAMPLES` | `1` | Grade N times and report the spread |
| `[grading] workers` | `MAX_WORKERS` | `5` | Students graded in parallel |
| `[grading] max_image_mb` | `MAX_IMAGE_MB` | `18` | Page-image budget per submission; DPI steps down to fit |
| `[routing] min_chars_per_page` | `MIN_CHARS_PER_PAGE` | `400` | Below this a PDF is treated as handwritten |
| `[routing] min_answered_fraction` | `MIN_ANSWERED_FRACTION` | `0.25` | Splitter found fewer answers than this: vision |
| `[students] id_pattern` | `STUDENT_ID_PATTERN` | none | Regex on the filename; group `id` is the student id |
| `[paths] submissions` | `PDF_FOLDER` | `submissions` | Student PDFs, or one subfolder per assignment |
| `[paths] rubrics` | `ANSWERS_FOLDER` | `rubrics` | Marking schemes |
| `[paths] results` | `RESULTS_FOLDER` | `results` | Grades, review state, ledger |
| `[paths] extracted` | `EXTRACTED_FOLDER` | `.work/extracted` | Text route scratch |
| `[paths] questions` | `QUESTIONS_FOLDER` | `.work/questions` | Text route scratch |
| `[paths] postures`, `tagsets`, `locales` | | same names | Your own data files; they override shipped ones by name |

## Per question

```toml
[[questions]]
id = 2
points = {a = 5, b = 15}   # or a number, split evenly over the parts
parts = ["a", "b"]         # used when --parts is not given
rubric = "rubrics/sql.md"  # default: rubrics/q2.md, then rubrics/correct_answer_2.md
```

Models always grade 0-100. `points` sets the weight of each item in the review
console and the marks total in the CSV export. Weights changed in the console
win over `points`.

## Student ids

The student id is the PDF's filename without `.pdf`. If your LMS names files
like `HW3_402100001_attempt2.pdf`, set:

```toml
[students]
id_pattern = "(?P<id>\\d{6,})"
```

A filename the pattern does not match keeps its whole name as the id, so it
never merges into another student.

## Postures and tag schemes

Shipped postures: `lenient`, `standard`, `strict`. Shipped tag schemes:
`sql_constructs`, `proof_technique`. To write your own, copy one from
`src/ta_grader/data/` into the workspace's `postures/` or `tagsets/` folder
and give it a new name. [design.md](design.md) explains what each may and may
not contain.
