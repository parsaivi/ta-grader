"""
Split an extracted homework into one file per question.

Design note
-----------
The original implementation asked the LLM to return the full text of every
answer as JSON.  That meant the model had to reproduce the entire homework --
often 20-80 KB -- inside a 4096-token reply.  Long submissions were truncated
mid-JSON, parsing failed, and the error handler quietly returned empty strings
for every question.  The result was a directory full of "[NO ANSWER PROVIDED]"
files for students who had in fact answered everything.

Instead we now number the lines of the homework and ask the model for nothing
but the line number where each answer starts.  The reply is a few dozen bytes
regardless of homework size, so it cannot truncate, and the answer text is
sliced locally from the original -- meaning it is always byte-for-byte what the
student wrote, with no risk of the model paraphrasing or dropping content.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

from . import locales
from .config import NO_ANSWER_MARKER, SPLITTER_MODEL, get_client

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


_SPLITTER_PROMPT_TEMPLATE = """You locate where each question's answer begins in a student's homework.

You receive the homework as numbered lines, like:

     1| Name: ...
     2| Question 1
     3| SRAM is built from flip-flops...

Students mark questions in many ways:
{examples}
- Sub-parts like {subparts} belong to the question above them,
  they are NOT new questions.

Your job:
For each question from 1 to N, report the line number where that question's
answer STARTS. The start line is the line containing the question heading
itself (or, if there is no heading, the first line of the answer's content).

Rules:
- Line numbers must strictly increase: question 2 starts after question 1.
- If a question was clearly not answered at all, report null for it.
- Do not report the line numbers of the printed question text if the student
  copied the questions; prefer where their own answer begins.
- Ignore cover pages, names, and student IDs.

Output STRICT JSON and nothing else:
{{"starts": {{"1": 3, "2": 27, "3": null, "4": 61}}}}
"""


def splitter_system_prompt() -> str:
    """The splitter's prompt, with heading examples from the enabled locales."""
    locs = locales.active()
    examples = [f'- "{e}"' for e in ["Question 1:", "Q1:", "1.", "1)", "1-", "#1"]]
    for loc in locs.locales:
        if loc.heading_examples and loc.code != "en":
            shown = ", ".join(f'"{e}"' for e in loc.heading_examples)
            examples.append(f"- {loc.name}: {shown}")
    subparts = ['"(a)"', '"(b)"'] + [f'"{x}"' for x in list(locs.part_label_map())[:3]]
    return _SPLITTER_PROMPT_TEMPLATE.format(
        examples="\n".join(examples), subparts=", ".join(subparts)
    )


# Question headings come from the enabled locale packs (locales.py): the words
# students use ("Question", "سوال"), their digits, and for right-to-left
# languages the number-first form PDFs extract ("۱ سوال").
def _keyword_heading_re():
    return locales.heading_regex(locales.active())


# A bare number used as a heading ("3", "3.", "3)"). Ambiguous -- it also
# matches ordinary numbered lists like "3. **Bottom diagram:** ..." -- so this
# only counts when the number stands alone on its line.
_BARE_HEADING_RE = re.compile(r"^\s*(\d{1,2})\s*[.):\-]?\s*$")

# Vision-described figures are prose written by the model, not by the student,
# and they are full of numbered lists. Never look for headings inside them.
_IMAGE_BLOCK_RE = re.compile(r"^\s*\[(IMAGE|TABLE)\b", re.IGNORECASE)


def _normalise_digits(text: str) -> str:
    return text.translate(locales.active().digit_table())


def number_lines(text: str) -> str:
    """Prefix each line with its 1-based line number."""
    lines = text.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}| {line}" for i, line in enumerate(lines, 1))


def find_headings_heuristically(text: str, num_questions: int) -> Dict[int, int]:
    """
    Find question headings with a regex, without calling the LLM.

    Returns {question_number: line_number}.  Used as a fallback when the LLM
    is unavailable, and as a sanity check on the LLM's answer.
    """
    found: Dict[int, int] = {}
    in_image_block = False
    keyword_re = _keyword_heading_re()

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = _normalise_digits(raw_line)

        # An [IMAGE DESCRIPTION] block runs until the next blank line.
        if _IMAGE_BLOCK_RE.match(line):
            in_image_block = True
            continue
        if in_image_block:
            if not line.strip():
                in_image_block = False
            continue

        match = keyword_re.match(line) or _BARE_HEADING_RE.match(line)
        if not match:
            continue

        number = next(g for g in match.groups() if g is not None)
        q = int(number)
        if not 1 <= q <= num_questions:
            continue

        # Keep the first plausible occurrence of each question number, and only
        # if it comes after the previous question's heading.
        if q in found:
            continue
        if found and line_no <= max(found.values()):
            continue
        found[q] = line_no

    return found


def _sanitise_starts(
    starts: Dict[int, Optional[int]],
    num_questions: int,
    total_lines: int,
) -> Dict[int, Optional[int]]:
    """
    Drop start lines that are out of range or out of order.

    A model that hallucinates a decreasing line number would otherwise produce
    an empty or negative-length slice, silently losing an answer.
    """
    clean: Dict[int, Optional[int]] = {}
    previous = 0

    for q in range(1, num_questions + 1):
        value = starts.get(q)
        if value is None:
            clean[q] = None
            continue
        if not isinstance(value, int) or not 1 <= value <= total_lines:
            clean[q] = None
            continue
        if value <= previous:
            clean[q] = None
            continue
        clean[q] = value
        previous = value

    return clean


def slice_by_starts(
    text: str,
    starts: Dict[int, Optional[int]],
    num_questions: int,
) -> Dict[str, str]:
    """
    Cut the homework into per-question text using the start line numbers.

    Each question runs from its start line up to the start of the next
    answered question (or end of file for the last one).
    """
    lines = text.splitlines()
    result: Dict[str, str] = {}

    # The boundary for question q is the next question that actually has a
    # start line, so an unanswered question in the middle does not truncate
    # the answer before it.
    for q in range(1, num_questions + 1):
        start = starts.get(q)
        if start is None:
            result[str(q)] = ""
            continue

        end = len(lines)
        for later in range(q + 1, num_questions + 1):
            later_start = starts.get(later)
            if later_start is not None:
                end = later_start - 1
                break

        result[str(q)] = "\n".join(lines[start - 1 : end]).strip()

    return result


class QuestionSplitter:
    """Works out where each question's answer begins."""

    def __init__(self, model: str = SPLITTER_MODEL):
        self.client = get_client()
        self.model = model

    def find_starts(self, text: str, num_questions: int) -> Dict[int, Optional[int]]:
        """Ask the LLM for the start line of each question."""
        numbered = number_lines(text)
        total_lines = len(text.splitlines())

        user_prompt = (
            f"Number of questions: {num_questions}\n"
            f"The homework has {total_lines} lines.\n\n"
            f"Homework:\n{numbered}\n\n"
            f"Report the starting line number for each of the {num_questions} questions."
        )

        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": splitter_system_prompt()},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    # The reply is only a handful of numbers, so this is ample.
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                parsed = self._parse_starts(content, num_questions)
                if parsed is not None:
                    return parsed
                last_error = ValueError(f"could not parse response: {content!r:.200}")
            except Exception as exc:  # noqa: BLE001 - retried and reported below
                last_error = exc

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

        raise RuntimeError(f"splitter failed after {MAX_RETRIES} attempts: {last_error}")

    @staticmethod
    def _parse_starts(
        text: Optional[str],
        num_questions: int,
    ) -> Optional[Dict[int, Optional[int]]]:
        """Parse the model's JSON reply into {question: start_line}."""
        if not text:
            return None

        match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            candidate = match.group(1)
        else:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            candidate = match.group(0)

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        raw = data.get("starts", data)
        if not isinstance(raw, dict):
            return None

        starts: Dict[int, Optional[int]] = {}
        for q in range(1, num_questions + 1):
            value = raw.get(str(q), raw.get(q))
            if isinstance(value, bool):
                value = None
            elif isinstance(value, float):
                value = int(value)
            elif isinstance(value, str):
                value = int(value) if value.strip().isdigit() else None
            elif not isinstance(value, int):
                value = None
            starts[q] = value

        return starts

    def split_homework(
        self,
        text: str,
        num_questions: int,
        use_llm: bool = True,
    ) -> Dict[str, str]:
        """
        Split homework text into {question_number: answer_text}.

        Falls back to regex heading detection if the LLM cannot be reached, so
        a network failure degrades to a rough split rather than losing the
        student's work entirely.
        """
        total_lines = len(text.splitlines())
        starts: Dict[int, Optional[int]] = {}

        if use_llm:
            try:
                starts = self.find_starts(text, num_questions)
            except Exception as exc:  # noqa: BLE001
                print(f"  LLM split failed ({exc}); falling back to regex headings")

        if not any(v is not None for v in starts.values()):
            starts = {
                q: line
                for q, line in find_headings_heuristically(text, num_questions).items()
            }

        starts = _sanitise_starts(starts, num_questions, total_lines)
        return slice_by_starts(text, starts, num_questions)


def split_all_homeworks(
    input_folder: str,
    output_base_folder: str,
    num_questions: int,
    use_llm: bool = True,
    overwrite: bool = True,
) -> Dict[str, int]:
    """
    Split every extracted homework into per-question files.

    Creates:
        output_base_folder/question_1/<student_id>.txt
        output_base_folder/question_2/<student_id>.txt
        ...
    """
    input_path = Path(input_folder)
    output_path = Path(output_base_folder)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Extracted-text folder not found: {input_folder}\n"
            f"Run the extract step first."
        )

    for q in range(1, num_questions + 1):
        (output_path / f"question_{q}").mkdir(parents=True, exist_ok=True)

    splitter = QuestionSplitter() if use_llm else None
    txt_files = sorted(input_path.glob("*.txt"))
    print(f"Found {len(txt_files)} homework files to split")

    stats = {"processed": 0, "failed": 0, "empty_answers": 0}

    for txt_file in txt_files:
        student_id = txt_file.stem
        try:
            content = txt_file.read_text(encoding="utf-8")

            if splitter is not None:
                questions = splitter.split_homework(content, num_questions, use_llm=True)
            else:
                found = find_headings_heuristically(content, num_questions)
                clean = _sanitise_starts(
                    dict(found), num_questions, len(content.splitlines())
                )
                questions = slice_by_starts(content, clean, num_questions)

            answered = 0
            for q_num, answer in questions.items():
                target = output_path / f"question_{q_num}" / f"{student_id}.txt"
                if not overwrite and target.exists():
                    continue
                if answer:
                    answered += 1
                else:
                    stats["empty_answers"] += 1
                target.write_text(answer or NO_ANSWER_MARKER, encoding="utf-8")

            stats["processed"] += 1
            print(f"  {student_id}: {answered}/{num_questions} questions found")

            # Surface the failure mode that used to pass silently.
            if answered == 0:
                print(
                    f"    WARNING: no answers located for {student_id} "
                    f"({len(content)} chars of text). Check the extraction output."
                )

        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            print(f"  {student_id}: ERROR {exc}")

    print(
        f"\nSplit complete: {stats['processed']} processed, "
        f"{stats['failed']} failed, {stats['empty_answers']} unanswered slots"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Split extracted homeworks into per-question files"
    )
    parser.add_argument("--input", "-i", default="students_extracted",
                        help="Folder of extracted .txt homeworks")
    parser.add_argument("--output", "-o", default="questions",
                        help="Folder to write question_N/ subfolders into")
    parser.add_argument("--num-questions", "-n", type=int, required=True,
                        help="Number of questions in the homework")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use only regex heading detection (no API calls)")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Keep existing per-question files")
    parser.add_argument("--preview", metavar="FILE",
                        help="Split a single file and print the result without saving")
    args = parser.parse_args()

    if args.preview:
        text = Path(args.preview).read_text(encoding="utf-8")
        splitter = QuestionSplitter() if not args.no_llm else None
        if splitter is not None:
            questions = splitter.split_homework(text, args.num_questions)
        else:
            found = find_headings_heuristically(text, args.num_questions)
            clean = _sanitise_starts(
                dict(found), args.num_questions, len(text.splitlines())
            )
            questions = slice_by_starts(text, clean, args.num_questions)

        for q in range(1, args.num_questions + 1):
            answer = questions.get(str(q), "")
            preview = answer[:300].replace("\n", " ")
            print(f"\n--- Q{q} ({len(answer)} chars) ---\n{preview or NO_ANSWER_MARKER}")
        return

    split_all_homeworks(
        input_folder=args.input,
        output_base_folder=args.output,
        num_questions=args.num_questions,
        use_llm=not args.no_llm,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
