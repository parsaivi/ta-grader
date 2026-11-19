"""
`ta-grader demo` -- a synthetic course workspace to try the tool on.

Builds a workspace with one two-part question, a marking scheme, and seven
made-up students: six typed submissions of varying quality (one skips part
(b)), and one handwritten-style submission with no text layer, so routing
sends it to the vision grader. Every name and id is invented.

Nothing here calls a model. `ta-grader route -q 1` on the result is free.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pymupdf as fitz

QUESTION = """\\question
Question 1. An algorithm sorts n numbers by repeatedly finding the smallest
remaining number and moving it to the front.

(a) What is its worst-case running time? Explain why.
(b) Is it stable? Justify your answer or give a counterexample.

\\answer
(a) O(n^2). Finding the minimum of the k remaining items takes k-1
comparisons, and this happens for k = n, n-1, ..., 2, so the total is
n(n-1)/2 comparisons, which is Theta(n^2) in every case, not only the worst.

Marking for (a), out of 100:
- 50: states O(n^2) (or Theta(n^2))
- 50: justifies it with the sum of comparisons, or an equivalent argument

(b) Not stable as usually implemented (swap with the front element).
Counterexample: [2a, 2b, 1] -> swapping 1 with 2a gives [1, 2b, 2a], so the
two 2s have changed order. It can be made stable by shifting instead of
swapping.

Marking for (b), out of 100:
- 40: says it is not stable
- 60: a correct counterexample or argument
- An answer that says "stable if implemented by shifting" with that
  justification also gets full marks.
"""

# (student id, answer to (a), answer to (b) or None for skipped)
TYPED: List[Tuple[str, str, str]] = [
    ("S1001",
     "The running time is O(n^2). Each pass scans the remaining elements to find "
     "the minimum: n-1 comparisons, then n-2, and so on down to 1. The sum is "
     "n(n-1)/2, which is quadratic.",
     "It is not stable. Take [2a, 2b, 1]. The first pass swaps 1 with 2a, giving "
     "[1, 2b, 2a], so 2a now comes after 2b."),
    ("S1002",
     "O(n^2) because there are two nested loops.",
     "Not stable, because it uses swaps."),
    ("S1003",
     "O(n log n), since each step halves the remaining work.",
     "It is stable because equal elements are never compared."),
    ("S1004",
     "Worst case is O(n^2): the inner loop does n-i comparisons for i = 1..n-1, "
     "total n(n-1)/2. Actually this is the cost for every input, not only the "
     "worst one.",
     "If we shift elements right instead of swapping, the minimum is inserted "
     "before equal keys never move past each other, so that version is stable. "
     "The swap version is not: [3a, 3b, 1] becomes [1, 3b, 3a]."),
    ("S1005",
     "Theta(n^2). Sum of (n-1) + (n-2) + ... + 1 comparisons.",
     None),
    ("S1006",
     "O(n). We go through the list once and move the smallest to the front.",
     "Yes it is stable."),
]

HANDWRITTEN = ("S1007",
               "O(n^2): finding the min takes n-1, n-2, ... comparisons, sum = n(n-1)/2.",
               "Not stable: [2a, 2b, 1] -> [1, 2b, 2a].")

COURSE = """\
# A synthetic demo course. Every student here is invented.

[course]
name = "Demo: Algorithms"
languages = ["en"]

[api]
url = "https://openrouter.ai/api/v1"

[models]
grader_1 = "openai/gpt-4.1-nano"
grader_2 = "google/gemini-2.5-flash"
splitter = "google/gemini-2.5-flash"
figures = "google/gemini-2.5-flash"

[grading]
conflict_limit = 10
resolution = "max"
posture = "standard"

[paths]
submissions = "submissions"
rubrics = "rubrics"
results = "results"
extracted = ".work/extracted"
questions = ".work/questions"

[[questions]]
id = 1
points = {a = 6, b = 4}
parts = ["a", "b"]
"""


def _typed_page(doc: fitz.Document, student_id: str, a: str, b: str | None) -> None:
    page = doc.new_page()
    y = 72
    page.insert_text((72, y), f"Homework 1          Student {student_id}", fontsize=12)
    y += 36
    body = f"Question 1\n\n(a) {a}\n\n" + (f"(b) {b}\n" if b else "")
    # Padding text so the page clears the typed-PDF threshold the way a real
    # page of homework does (routing treats < 400 chars/page as handwritten).
    body += ("\nNotes: I checked my answers against the lecture slides on "
             "selection sort, and wrote out the comparison counts by hand for "
             "n = 4 to make sure the sum is right: 3 + 2 + 1 = 6 = 4*3/2. "
             "I also traced the algorithm on the list [5, 3, 8, 1] from the "
             "tutorial: after the first pass it is [1, 3, 8, 5], after the "
             "second [1, 3, 8, 5], and after the third [1, 3, 5, 8]. Each pass "
             "fixes one more position at the front of the list.\n")
    rect = fitz.Rect(72, y, page.rect.width - 72, page.rect.height - 72)
    page.insert_textbox(rect, body, fontsize=11, fontname="helv")


def _image_only_page(doc: fitz.Document, student_id: str, a: str, b: str) -> None:
    """Render text to a picture and place only the picture: no text layer."""
    scratch = fitz.open()
    _typed_page(scratch, student_id, a, b)
    pix = scratch[0].get_pixmap(dpi=110)
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)


def build(target: Path) -> Path:
    target = target.resolve()
    (target / "submissions").mkdir(parents=True, exist_ok=True)
    (target / "rubrics").mkdir(exist_ok=True)
    (target / "results").mkdir(exist_ok=True)
    (target / "course.toml").write_text(COURSE, encoding="utf-8")
    (target / "rubrics" / "q1.md").write_text(QUESTION, encoding="utf-8")
    (target / ".gitignore").write_text(".env\nresults/\n.work/\n", encoding="utf-8")
    if not (target / ".env").exists():
        (target / ".env").write_text("LLM_KEY=\n", encoding="utf-8")

    for student_id, a, b in TYPED:
        doc = fitz.open()
        _typed_page(doc, student_id, a, b)
        doc.save(target / "submissions" / f"{student_id}.pdf")

    student_id, a, b = HANDWRITTEN
    doc = fitz.open()
    _image_only_page(doc, student_id, a, b)
    doc.save(target / "submissions" / f"{student_id}.pdf")
    return target
