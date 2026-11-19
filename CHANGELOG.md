# Changelog

## 0.1.0 (unreleased)

First public release.

- One `ta-grader` command: `init`, `demo`, `route`, `grade`, `review`,
  `export`, `benchmark`, `models`.
- Course workspaces with `course.toml`, kept separate from the code.
- Locale packs for question headings, digits, part labels and right-to-left
  repair. English and Persian ship.
- Per-question parts, points and rubric files.
- Configurable mapping from PDF filename to student id.
- `ta-grader export` refuses a batch nobody has signed off.
- A synthetic demo course and an offline end-to-end test.
- Fixed: the PDF extractors ignored the configured endpoint and figure model.
