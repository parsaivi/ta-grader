# Responsible use

ta-grader drafts grades. A TA decides them. The tool is built so that this is
a step someone has to take: export is blocked until a person signs off the
batch, and every change a reviewer makes is logged beside the model's
original grade.

## Before you use it on a real class

- **Check the models on your own marking.** Grade a handful of submissions
  yourself, then run `ta-grader benchmark` on them. A model that cannot read
  your students' language or handwriting often returns 0 instead of saying
  so.
- **Tell your instructor and, where your institution expects it, your
  students** that an AI model drafts grades and a TA reviews them.
- **Keep the posture fixed for the whole round.** Grading half a class under
  `strict` and half under `lenient` is unfair by construction. Changing it
  means regrading everyone.
- **Read the flagged items, not just the totals.** The review queue exists
  because both models can be wrong in the same direction.

## Student data

Grading sends images or text of each submission to the model provider you
configure. Before you do that:

- Find out whether your institution allows student work to be sent to that
  provider. Rules such as FERPA (US) and GDPR (EU) may apply. A provider with
  a no-training, no-retention agreement, or a model you host yourself
  (vLLM, Ollama), avoids most of the question.
- Remove names from PDFs if you can; the filename only needs an id.
- Keep the workspace private. `submissions/`, `results/` and `.env` are in the
  workspace's `.gitignore` by default. The review ledger
  (`*.review.jsonl`) is the most sensitive file: it holds ids, marks, the
  reviewer's corrections and verbatim quotes from each student's work.
- The review console listens on localhost only.

## What it is not for

- Final grades with no human review.
- High-stakes exams where a single wrong mark has serious consequences,
  unless every item is reviewed.
- Judging writing style, effort or honesty. It grades against the marking
  scheme you give it, and nothing else.
