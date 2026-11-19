"""
Vision-based homework grader.

Sends the student's PDF pages straight to two vision LLMs and compares their
grades.  Where the two models disagree by more than GRADE_CONFLICT_LIMIT, the
submission is flagged for human review in conflict_reviewer_app.py.

Both models really do run: an earlier version had model 1 commented out and
hardcoded `conflicts = [False] * n`, which meant every grade came from a single
un-cross-checked model and the conflict reviewer never had anything to show.
"""

import argparse
import base64
import concurrent.futures
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf as fitz
from tqdm import tqdm

from . import locales
from .assurance import (
    EVIDENCE_FIELD,
    LEGIBILITY_ALIASES,
    LEGIBILITY_FIELD,
    LEGIBILITY_VALUES,
    PRESENCE_ALIASES,
    PRESENCE_FIELD,
    PRESENCE_VALUES,
    PartAssurance,
    apply_vetoes,
    check_evidence,
    summarise_all_samples,
)
from .assurance import output_spec as assurance_output_spec
from .assurance import prompt_section as assurance_prompt_section
from .config import (
    ANSWERS_FOLDER,
    GRADE_CONFLICT_LIMIT,
    GRADE_RESOLUTION,
    GRADE_SAMPLES,
    MAX_IMAGE_MB,
    MAX_WORKERS,
    MODEL_1_NAME,
    MODEL_2_NAME,
    OUTPUT_FILE,
    PDF_FOLDER,
    POSTURE,
    POSTURES_FOLDER,
    QUESTION_PARTS,
    TAG_SCHEME,
    TAGSETS_FOLDER,
    get_client,
    result_path,
    rubric_file,
    question_parts,
    student_id_for,
)
from .grade_parsing import (
    extract_evidence,
    extract_grades,
    extract_single_grade,
    extract_tags,
    resolve_grade,
    resolve_grades,
)
from .posture import GradingPosture, default_posture, list_postures, load_posture
from .tagging import TagScheme, list_tag_schemes, load_tag_scheme

# Rendering resolution for PDF pages sent to the models. Higher is more legible
# for handwriting but costs more tokens.
DEFAULT_DPI = 150

# Floor for the automatic downscale; below this handwriting stops being legible
# and there is no point spending the call.
MIN_DPI = 60

# JPEG quality for rendered pages. 80 keeps handwriting crisp at roughly a
# fifth of PNG's size.
JPEG_QUALITY = 80

# Total rendered bytes allowed per submission. The API rejects requests whose
# images exceed 30 MB; this leaves headroom for base64 overhead and the prompt.
MAX_IMAGE_BYTES = int(MAX_IMAGE_MB * 1_000_000)


def _language_line() -> str:
    return locales.active().language_sentence()


def get_grading_prompt_single(question_num: int, correct_answer: str) -> str:
    """System prompt for grading one question out of a whole submission."""
    return f"""You are a strict but fair teaching assistant grading student homework.

You will see images of a student's homework submission (PDF pages).
You need to find and grade ONLY Question {question_num}.

{_language_line()} It may be handwritten. Grade the content, not the
handwriting or the language.

### Question {question_num} - Correct Answer:
{correct_answer}

### Grading Criteria:
- **100**: Perfect answer, fully correct
- **80-99**: Minor errors or missing small details
- **60-79**: Partially correct, major concepts understood but some errors
- **40-59**: Shows some understanding but significant errors
- **20-39**: Minimal understanding, mostly incorrect
- **0-19**: Completely wrong or no attempt

### Your Task:
1. Find the student's answer to Question {question_num} in the PDF
2. Compare it to the correct solution
3. Note what is correct and what is wrong
4. Assign a grade from 0 to 100

If you genuinely cannot find any answer to Question {question_num}, say so
explicitly and grade it 0.

### Output Format:
First, provide brief reasoning for your grade.
Then end your response with the grade on its own line, in exactly this format:

GRADE: 85

Write the number only. Do not use brackets, ranges, or "/100"."""


def get_grading_prompt_all(num_questions: int, correct_answers: str) -> str:
    """System prompt for grading every question in one pass."""
    return f"""You are a strict but fair teaching assistant grading student homework.

You will see images of a student's homework submission (PDF pages).
The homework has {num_questions} questions.

{_language_line()} It may be handwritten. Grade the content, not the
handwriting or the language.

### Correct Answers:
{correct_answers}

### Grading Criteria:
- **100**: Perfect answer, fully correct
- **80-99**: Minor errors or missing small details
- **60-79**: Partially correct, major concepts understood but some errors
- **40-59**: Shows some understanding but significant errors
- **20-39**: Minimal understanding, mostly incorrect
- **0-19**: Completely wrong or no attempt

### Your Task:
1. Examine each page of the student's submission
2. Identify answers for each of the {num_questions} questions
3. Compare each answer to the correct solution
4. Grade each question from 0 to 100

A question the student did not attempt gets 0.

### Output Format:
First, provide brief reasoning for each question's grade.
Then end your response with the grades on their own line, in exactly this format:

GRADES: [85, 70, 100, 45, 90, 60, 75, 80]

The array MUST contain exactly {num_questions} numbers, in question order.
Write numbers only -- no ranges, no "/100", no extra text inside the brackets."""


def get_grading_prompt_parts(
    question_num: int,
    part_labels: List[str],
    correct_answer: str,
    tag_scheme: "TagScheme" = None,
    posture: "GradingPosture" = None,
) -> str:
    """
    System prompt for grading one question that is marked part by part.

    Each part gets its own independent 0-100 grade. This is not the same as
    grading N questions: the parts share one question statement and one marking
    scheme, and a student's answer to all of them usually sits in one block of
    the page, so the model needs to see the whole thing at once and then split
    its judgement.

    `tag_scheme` optionally adds a course-specific label per part (see
    tagging.py). Without one the prompt never mentions labels at all -- any
    rule about what students may or may not use belongs in the marking scheme
    file, not in this builder.

    `posture` states the marginal-call rules (see posture.py). Unlike the tag
    scheme it is never omitted: a prompt that says nothing about how to treat a
    borderline answer does not get neutrality, it gets whatever the model felt
    like that time.
    """
    listed = ", ".join(f"({p})" for p in part_labels)
    n = len(part_labels)
    example = ", ".join(str(v) for v in ([85, 70, 40] * n)[:n])
    posture = posture or default_posture()

    tag_task = f"\n5. {tag_scheme.instruction}" if tag_scheme and tag_scheme.instruction else ""
    tag_block = f"\n{tag_scheme.prompt_section('part')}\n" if tag_scheme else ""
    tag_reasoning = (
        " State the label you are assigning and what in the answer justifies it."
        if tag_scheme else ""
    )
    tag_output = (
        "\n" + tag_scheme.output_spec(n, f"part order ({listed})")
        if tag_scheme else ""
    )

    return f"""You are a strict but fair teaching assistant grading student homework.

You will see images of a student's homework submission (PDF pages).
You must find and grade ONLY Question {question_num}.

Question {question_num} has {n} parts: {listed}. Grade each part SEPARATELY and
independently. A student may answer some parts and not others.

{_language_line()} It may be handwritten. Grade the content, not the
handwriting or the language. {locales.active().part_label_sentence(part_labels)}
Students may also label parts a/b/c, 1/2/3, or i/ii/iii, or not label them at
all -- in that case work out which part each answer belongs to from its content.

### Question {question_num} - Statement, reference solution and marking scheme:
{correct_answer}

{posture.prompt_section()}

### Your Task:
1. Locate the student's answer to Question {question_num} across the pages.
2. For each part, decide which part of the question it answers.
3. For each part, report whether you can read it and whether it is there at all,
   and quote the line your grade turns on.
4. Compare each part to the reference solution and the marking bands, and
   assign it a grade from 0 to 100, applying the marking scheme above and
   nothing else.{tag_task}

A part the student attempted and left wrong gets the grade the marking scheme
says. A part that is blank, unreadable, or absent from these pages is NOT graded
0 -- report it in the fields below and leave its grade as 0 in the array; the
grading system will decide what to record. Do not invent a grade for work you
could not see.

{assurance_prompt_section('part')}
{tag_block}
### Output Format:
First, write a short paragraph per part: what you found, what is correct and
what is wrong.{tag_reasoning}

Then end your response with the following, in exactly this format:

GRADES: [{example}]

The GRADES array MUST contain exactly {n} numbers, in part order ({listed}).
Write numbers only -- no ranges, no "/100", no letters.

{assurance_output_spec(n, f"part order ({listed})")}
{tag_output}"""


@dataclass
class GradingResult:
    model: str
    grades: List[Optional[int]]
    raw_output: str


@dataclass
class CallUsage:
    """Token counts and cost for one model call."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    failed: bool = False


@dataclass
class UsageLedger:
    """
    Thread-safe accumulator for what a run actually cost.

    Grading calls run across a thread pool, so every mutation is under a lock.
    Cost comes from the provider's own accounting (OpenRouter returns it when
    asked for) rather than being computed from a hardcoded price table, which
    goes stale silently and is wrong for image tokens in particular.
    """

    calls: List[CallUsage] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, usage: CallUsage) -> None:
        with self._lock:
            self.calls.append(usage)

    def by_model(self) -> Dict[str, Dict[str, float]]:
        totals: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for c in self.calls:
                t = totals.setdefault(
                    c.model,
                    {"calls": 0, "failed": 0, "prompt_tokens": 0,
                     "completion_tokens": 0, "cost": 0.0},
                )
                t["calls"] += 1
                t["failed"] += int(c.failed)
                t["prompt_tokens"] += c.prompt_tokens
                t["completion_tokens"] += c.completion_tokens
                t["cost"] += c.cost
        return totals

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(c.cost for c in self.calls)


class VisionGrader:
    """Grades homework PDFs using two vision LLMs."""

    def __init__(
        self,
        model_1: str = MODEL_1_NAME,
        model_2: str = MODEL_2_NAME,
        dpi: int = DEFAULT_DPI,
        conflict_limit: int = GRADE_CONFLICT_LIMIT,
        resolution: str = GRADE_RESOLUTION,
        single: bool = False,
        tag_scheme: TagScheme = None,
        posture: GradingPosture = None,
        samples: int = 1,
        reference: str = "",
    ):
        self.client = get_client()
        self.model_1 = model_1
        self.model_2 = model_2
        # Opt-in single-model grading: only model_1 runs, halving cost and
        # giving up cross-checking. This is deliberately an explicit flag and
        # is recorded in the output -- a previous version silently ran one
        # model while still reporting results as dual-model verified.
        self.single = single
        self.dpi = dpi
        self.conflict_limit = conflict_limit
        self.resolution = resolution
        # Optional course-specific labels recorded next to each grade. None
        # means no tagging: no prompt section, no tag keys in the results.
        self.tag_scheme = tag_scheme
        # The marginal-call rules. Never None -- see posture.default_posture().
        self.posture = posture or default_posture()
        # Repeated gradings of the same submission, used to measure how much the
        # model agrees with itself. 1 means no sampling and no extra cost.
        self.samples = max(1, samples)
        # The marking scheme text, kept so evidence quotes can be checked
        # against it: a quote lifted from here rather than from the student is
        # the signature of a model reciting the answer key.
        self.reference = reference
        # Every call's tokens and cost land here so a run can report what it
        # actually spent instead of an estimate.
        self.usage = UsageLedger()

    def _render(self, doc, dpi: int) -> Tuple[List[bytes], int]:
        """Render every page at the given DPI as JPEG. Returns (pages, bytes)."""
        matrix = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is PDF's native DPI
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            # JPEG rather than PNG: these are photographs of paper, where PNG
            # is several times larger for no readability gain.
            pages.append(pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY))
        return pages, sum(len(p) for p in pages)

    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Render every PDF page to a base64-encoded JPEG, within a size budget.

        Scanned submissions are the ones that must go through vision, and they
        are also the largest: a 17-page scan at 150 DPI exceeds the API's 30 MB
        image limit and the request is rejected outright. So the DPI is stepped
        down until the payload fits, rather than failing the student.
        """
        doc = fitz.open(pdf_path)
        try:
            dpi = self.dpi
            pages, total = self._render(doc, dpi)

            while total > MAX_IMAGE_BYTES and dpi > MIN_DPI:
                dpi = max(MIN_DPI, int(dpi * 0.75))
                pages, total = self._render(doc, dpi)

            if total > MAX_IMAGE_BYTES:
                print(f"Warning: {Path(pdf_path).name} is {total / 1e6:.1f} MB even at "
                      f"{dpi} DPI ({len(pages)} pages); the request may be rejected.")
            elif dpi != self.dpi:
                print(f"  {Path(pdf_path).name}: reduced to {dpi} DPI "
                      f"({total / 1e6:.1f} MB, {len(pages)} pages)")

            return [base64.b64encode(p).decode("utf-8") for p in pages]
        finally:
            # Close even if rendering throws, so a bad PDF cannot leak handles
            # across a few hundred submissions.
            doc.close()

    def call_vision_llm(self, model_name: str, system_prompt: str, images: List[str]) -> str:
        """Send the pages to one model. Returns "" if the call fails."""
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Grade this student's homework submission:"}
        ]
        for i, img_base64 in enumerate(images):
            content.append({"type": "text", "text": f"[Page {i + 1}]"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                }
            )

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=4000,
                # Ask the gateway to report what the call cost. Vision cost is
                # driven by image tokenization, which varies ~6x between model
                # families, so a local price-per-token estimate is not usable.
                extra_body={"usage": {"include": True}},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error calling {model_name}: {exc}")
            self.usage.record(CallUsage(model=model_name, failed=True))
            return ""

        # A response can arrive successfully but truncated a few words in --
        # the call does not raise, and the grade line never appears. Surface it
        # rather than letting it look like a model that declined to grade.
        choice = response.choices[0] if response.choices else None
        finish = getattr(choice, "finish_reason", None) if choice else None
        if finish not in (None, "stop"):
            print(f"  {model_name}: response ended early (finish_reason={finish})")

        usage = getattr(response, "usage", None)
        self.usage.record(
            CallUsage(
                model=model_name,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                # Gateways that do not return cost leave this at 0; the token
                # counts still let you see where the spend went.
                cost=float(getattr(usage, "cost", 0) or 0),
            )
        )
        return response.choices[0].message.content or ""

    def _run_both_models(
        self, system_prompt: str, images: List[str]
    ) -> Tuple[str, str]:
        """Call both models concurrently and return their raw responses."""
        if self.single:
            return self.call_vision_llm(self.model_1, system_prompt, images), ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(self.call_vision_llm, self.model_1, system_prompt, images)
            f2 = pool.submit(self.call_vision_llm, self.model_2, system_prompt, images)
            return f1.result(), f2.result()

    def _run_graders(
        self, system_prompt: str, images: List[str]
    ) -> Tuple[List[str], str]:
        """
        Call the graders and return (model 1's samples, model 2's response).

        Model 1 runs `self.samples` times so its self-agreement can be measured;
        model 2 runs once, because it is there to disagree with model 1, not to
        have its own variance measured. Everything runs concurrently -- these
        are latency-bound calls and a five-sample run should not take five times
        as long as a one-sample run.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.samples + 1
        ) as pool:
            f1 = [
                pool.submit(self.call_vision_llm, self.model_1, system_prompt, images)
                for _ in range(self.samples)
            ]
            f2 = (
                None if self.single
                else pool.submit(self.call_vision_llm, self.model_2, system_prompt, images)
            )
            return [f.result() for f in f1], (f2.result() if f2 else "")

    def _error_result_all(
        self, student_id: str, num_questions: int, error: str
    ) -> Dict[str, Any]:
        """
        Build a full-shaped result for a submission that could not be read.

        The shape must match a successful result exactly -- the summary printer
        and the Streamlit reviewer both index model_1/model_2 unconditionally,
        and a short dict used to crash them with a KeyError.
        """
        return {
            "student_id": student_id,
            "error": error,
            "final_grades": [None] * num_questions,
            "total_grade": 0,
            "conflicts": [True] * num_questions,
            "has_conflict": True,
            "model_1": {"name": self.model_1, "grades": [None] * num_questions,
                        "reasoning": f"[PDF could not be read: {error}]"},
            "model_2": {"name": self.model_2, "grades": [None] * num_questions,
                        "reasoning": f"[PDF could not be read: {error}]"},
        }

    def _error_result_single(
        self, student_id: str, question_num: int, error: str
    ) -> Dict[str, Any]:
        """Full-shaped result for a single-question grade that could not run."""
        return {
            "student_id": student_id,
            "question": question_num,
            "error": error,
            "final_grade": None,
            "conflict": True,
            "model_1": {"name": self.model_1, "grade": None,
                        "reasoning": f"[PDF could not be read: {error}]"},
            "model_2": {"name": self.model_2, "grade": None,
                        "reasoning": f"[PDF could not be read: {error}]"},
        }

    def grade_student(
        self, pdf_path: str, system_prompt: str, num_questions: int
    ) -> Dict[str, Any]:
        """Grade every question of one student's PDF with both models."""
        student_id = student_id_for(pdf_path)

        try:
            images = self.pdf_to_images(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Error reading PDF {pdf_path}: {exc}")
            return self._error_result_all(student_id, num_questions, str(exc))

        raw1, raw2 = self._run_both_models(system_prompt, images)
        grades1 = extract_grades(raw1, num_questions)
        grades2 = extract_grades(raw2, num_questions)

        final_grades, conflicts = resolve_grades(
            grades1, grades2, self.conflict_limit, self.resolution
        )

        return {
            "student_id": student_id,
            "final_grades": final_grades,
            "total_grade": sum(g for g in final_grades if g is not None),
            "conflicts": conflicts,
            "has_conflict": any(conflicts),
            "model_1": {"name": self.model_1, "grades": grades1, "reasoning": raw1},
            "model_2": {"name": self.model_2, "grades": grades2, "reasoning": raw2},
        }

    def grade_student_single_question(
        self, pdf_path: str, system_prompt: str, question_num: int
    ) -> Dict[str, Any]:
        """Grade one question of one student's PDF with both models."""
        student_id = student_id_for(pdf_path)

        try:
            images = self.pdf_to_images(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Error reading PDF {pdf_path}: {exc}")
            return self._error_result_single(student_id, question_num, str(exc))

        raw1, raw2 = self._run_both_models(system_prompt, images)
        grade1 = extract_single_grade(raw1)
        grade2 = extract_single_grade(raw2)

        final_grade, is_conflict = resolve_grade(
            grade1, grade2, self.conflict_limit, self.resolution
        )

        return {
            "student_id": student_id,
            "question": question_num,
            "final_grade": final_grade,
            "conflict": is_conflict,
            "model_1": {"name": self.model_1, "grade": grade1, "reasoning": raw1},
            "model_2": {"name": self.model_2, "grade": grade2, "reasoning": raw2},
        }

    def _error_result_parts(
        self, student_id: str, question_num: int, part_labels: List[str], error: str
    ) -> Dict[str, Any]:
        """Full-shaped per-part result for a submission that could not be read."""
        n = len(part_labels)
        return {
            "student_id": student_id,
            "question": question_num,
            "parts": part_labels,
            "single_model": self.single,
            "samples": self.samples,
            "posture": self.posture.key,
            "error": error,
            "part_grades": {p: None for p in part_labels},
            **self._tag_fields(part_labels, [None] * n),
            # Same shape as a successful result: the summary printers and the
            # Streamlit reviewer index these unconditionally, and a short dict
            # used to crash them at the end of a long, expensive run.
            "assurance": {
                p: PartAssurance(veto="unreadable_pdf").as_dict()
                for p in part_labels
            },
            "final_grades": [None] * n,
            "conflicts": [True] * n,
            "has_conflict": True,
            "model_1": {"name": self.model_1, "grades": [None] * n,
                        "reasoning": f"[PDF could not be read: {error}]"},
            "model_2": {"name": self.model_2, "grades": [None] * n,
                        "reasoning": f"[PDF could not be read: {error}]"},
        }

    def _tag_fields(
        self, labels: List[str], tags: List[Optional[str]]
    ) -> Dict[str, Any]:
        """
        The tag keys for a result dict, or nothing when no scheme is loaded.

        `tag_scheme` names which vocabulary produced the values, so a results
        file stays readable after the scheme changes: a run tagged with
        `constructs` and one tagged with `technique` are not silently merged.
        """
        if not self.tag_scheme:
            return {}
        return {
            "tag_scheme": self.tag_scheme.key,
            "part_tags": dict(zip(labels, tags)),
        }

    def grade_student_question_parts(
        self, pdf_path: str, system_prompt: str, question_num: int,
        part_labels: List[str],
    ) -> Dict[str, Any]:
        """
        Grade one question part by part, with both models.

        Each part is resolved and conflict-checked on its own: two models can
        agree on part (a) and disagree sharply on part (c), and collapsing that
        into a single question-level flag would hide exactly the disagreement a
        human needs to look at.
        """
        student_id = student_id_for(pdf_path)
        n = len(part_labels)

        try:
            images = self.pdf_to_images(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Error reading PDF {pdf_path}: {exc}")
            return self._error_result_parts(student_id, question_num, part_labels, str(exc))

        samples1, raw2 = self._run_graders(system_prompt, images)
        raw1 = samples1[0]

        # Model 1's grades, reduced across however many samples were taken. With
        # samples=1 this is the median of one value -- i.e. the value -- and the
        # spread is None, so the single-sample path behaves exactly as before.
        sample_grades = [extract_grades(s, n) for s in samples1]
        grades1, spreads, levels, per_part_samples = summarise_all_samples(
            sample_grades, n, self.conflict_limit
        )

        # Tags, legibility, presence and evidence all come from model 1's first
        # response only. They are observations about what is on the page, not
        # judgements to be cross-checked, and resolving two models' answers
        # would mean inventing a tie-break for a question of fact.
        tags = (
            extract_tags(raw1, n, self.tag_scheme.field, self.tag_scheme.names,
                         self.tag_scheme.aliases)
            if self.tag_scheme else [None] * n
        )
        legibility = extract_tags(
            raw1, n, LEGIBILITY_FIELD, LEGIBILITY_VALUES, LEGIBILITY_ALIASES
        )
        presence = extract_tags(
            raw1, n, PRESENCE_FIELD, PRESENCE_VALUES, PRESENCE_ALIASES
        )
        evidence = extract_evidence(raw1, n, EVIDENCE_FIELD)

        if self.single:
            grades2 = [None] * n
            final_grades = grades1
            # With one model there is nothing to disagree with, so the only
            # thing worth flagging is a part it failed to grade at all. That
            # stays None rather than becoming 0 -- "no grade" is not "zero".
            conflicts = [g is None for g in grades1]
        else:
            grades2 = extract_grades(raw2, n)
            final_grades, conflicts = resolve_grades(
                grades1, grades2, self.conflict_limit, self.resolution
            )

        # Withhold grades the model was not entitled to give: a part that is
        # absent from the pages or that it could not read. This runs last, after
        # resolution, because it overrides both models regardless of how well
        # they agreed -- two models confidently agreeing on a grade for a page
        # that is not in the file are both wrong.
        final_grades, veto_flags, vetoes = apply_vetoes(
            final_grades, legibility, presence, self.posture.unattempted
        )
        conflicts = [c or v for c, v in zip(conflicts, veto_flags)]

        evidence_ok, evidence_problems = check_evidence(
            evidence, final_grades, self.reference
        )
        if evidence_problems:
            print(f"  {student_id}: " + "; ".join(evidence_problems))

        # Confidence is not reported for a part whose grade was withheld. The
        # samples there agree tightly on a 0 the model was told to write for an
        # unreadable or absent answer, so "high confidence" would be true of a
        # number that is not a grade -- and it would inflate the run's
        # confidence tallies with parts that were never graded at all.
        assurance = [
            PartAssurance(
                legibility=legibility[i],
                presence=presence[i],
                evidence=evidence[i],
                evidence_ok=evidence_ok[i],
                veto=vetoes[i],
                confidence=(
                    levels[i] if self.samples > 1 and not vetoes[i] else None
                ),
                spread=None if vetoes[i] else spreads[i],
                samples=(
                    per_part_samples[i] if self.samples > 1 and not vetoes[i] else []
                ),
            ).as_dict()
            for i in range(n)
        ]

        return {
            "student_id": student_id,
            "question": question_num,
            "parts": part_labels,
            "single_model": self.single,
            "samples": self.samples,
            "posture": self.posture.key,
            # Keyed view for reading by hand; the list keeps the ordering that
            # resolve_grades and the summary printers work with.
            "part_grades": dict(zip(part_labels, final_grades)),
            # Descriptive only, and only when a tag scheme is loaded. Never
            # folded into the grade.
            **self._tag_fields(part_labels, tags),
            # Why each grade should or should not be trusted: readability,
            # presence, evidence, and self-agreement across samples.
            "assurance": dict(zip(part_labels, assurance)),
            "final_grades": final_grades,
            "conflicts": conflicts,
            "has_conflict": any(conflicts),
            "model_1": {
                "name": self.model_1,
                "grades": grades1,
                "reasoning": raw1,
                # Only kept when there is more than one, to avoid tripling the
                # size of every results file for no gain.
                **({"samples": samples1} if self.samples > 1 else {}),
            },
            "model_2": {
                "name": None if self.single else self.model_2,
                "grades": grades2,
                "reasoning": "[single-model run: second grader not called]"
                if self.single else raw2,
            },
        }


# --- Answer file loading ------------------------------------------------------


def parse_answer_file(content: str) -> Tuple[str, str]:
    r"""
    Split an answer file into (question, answer).

    Files may use \question and \answer markers; a file with neither is treated
    as being entirely the answer.
    """
    question = ""
    answer = content.strip()

    if "\\answer" in content:
        head, _, tail = content.partition("\\answer")
        answer = tail.strip()
        question = head.split("\\question", 1)[-1].strip() if "\\question" in head else head.strip()

    return question, answer


def load_correct_answer(answers_folder: str, question_num: int) -> Optional[str]:
    """
    Load the reference answer for one question.

    Returns None when the file is missing, so callers can report the specific
    path instead of sending the string "[No answer file found]" to the model as
    though it were the marking scheme.
    """
    filepath = rubric_file(question_num, answers_folder)
    if not filepath.exists():
        return None

    question, answer = parse_answer_file(filepath.read_text(encoding="utf-8"))
    return f"{question}\n\n{answer}".strip() if question else answer


def load_all_correct_answers(answers_folder: str, num_questions: int) -> str:
    """Concatenate every question's reference answer into one prompt block."""
    parts = []
    missing = []

    for q in range(1, num_questions + 1):
        content = load_correct_answer(answers_folder, q)
        if content is None:
            missing.append(q)
            content = "[No reference answer supplied for this question.]"
        parts.append(f"### Question {q}:\n{content}")

    if missing:
        print(
            f"Warning: no answer file for question(s) {missing}. "
            f"Expected {answers_folder}/q<N>.md or correct_answer_<N>.md"
        )

    return "\n\n".join(parts)


# --- Result persistence -------------------------------------------------------


def load_existing_results(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Load previously graded students so a run can resume where it stopped."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {r["student_id"]: r for r in data.get("grades", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        print(f"Warning: could not read {filepath}; starting fresh.")
        return {}


def save_results(
    results: List[Dict[str, Any]],
    filepath: str,
    num_questions: int = None,
    question_num: int = None,
    part_labels: List[str] = None,
    usage: "UsageLedger" = None,
    tag_scheme: TagScheme = None,
    posture: GradingPosture = None,
    samples: int = 1,
):
    """Write results atomically so an interrupt cannot leave truncated JSON."""
    if part_labels:
        output = {
            "question": question_num,
            "parts": part_labels,
            "total_students": len(results),
            "students_with_conflicts": sum(
                1 for r in results if r.get("has_conflict", False)
            ),
            "grades": results,
        }
    elif question_num:
        output = {
            "question": question_num,
            "total_students": len(results),
            "conflicts": sum(1 for r in results if r.get("conflict", False)),
            "grades": results,
        }
    else:
        output = {
            "num_questions": num_questions,
            "total_students": len(results),
            "students_with_conflicts": sum(
                1 for r in results if r.get("has_conflict", False)
            ),
            "grades": results,
        }

    if tag_scheme is not None:
        # Record the vocabulary, not just the labels: a file whose tags read
        # "cte"/"none" is unreadable a term later without knowing what asked.
        output["tag_scheme"] = {
            "key": tag_scheme.key,
            "title": tag_scheme.title,
            "labels": tag_scheme.names,
        }

    if posture is not None:
        # Posture changes grades, so the full set of choices is recorded, not
        # just its name: "strict" means whatever strict.json said on the day of
        # the run, and a file that only stored the name would silently
        # misdescribe its own marks after any edit to that file.
        output["posture"] = posture.as_dict()

    if samples and samples > 1:
        output["samples"] = samples

    if usage is not None:
        output["usage"] = {
            "total_cost_usd": round(usage.total_cost, 6),
            "by_model": {
                m: {**t, "cost": round(t["cost"], 6)}
                for m, t in usage.by_model().items()
            },
        }

    tmp = f"{filepath}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp, filepath)


# --- Drivers ------------------------------------------------------------------


def _collect_pdfs(pdf_folder: str) -> List[Path]:
    """Find student PDFs, case-insensitively on the extension."""
    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        print(f"Error: PDF folder '{pdf_folder}' not found!")
        return []

    pdfs = [p for p in sorted(pdf_path.iterdir())
            if p.is_file() and p.suffix.lower() == ".pdf"]
    print(f"Found {len(pdfs)} PDF submissions")
    return pdfs


def _run_grading(pdfs: List[Path], grade_fn, results: List[Dict[str, Any]], save_fn):
    """Grade a batch of PDFs in parallel, saving after each completion."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(grade_fn, pdf): student_id_for(pdf) for pdf in pdfs}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            student_id = futures[future]
            try:
                results.append(future.result())
                save_fn(results)
            except Exception as exc:  # noqa: BLE001
                print(f"Error grading {student_id}: {exc}")


def grade_single_question(
    question_num: int,
    pdf_folder: str = PDF_FOLDER,
    answers_folder: str = ANSWERS_FOLDER,
    output_file: str = None,
    grader: VisionGrader = None,
):
    """Grade one question for every student."""
    output_file = output_file or result_path(f"grades_question_{question_num}.json")

    print(f"Loading correct answer for question {question_num}...")
    correct_answer = load_correct_answer(answers_folder, question_num)
    if correct_answer is None:
        print(f"Error: {answers_folder}/correct_answer_{question_num}.md not found!")
        return

    system_prompt = get_grading_prompt_single(question_num, correct_answer)

    pdf_files = _collect_pdfs(pdf_folder)
    if not pdf_files:
        return

    existing = load_existing_results(output_file)
    to_grade = [p for p in pdf_files if student_id_for(p) not in existing]
    print(f"Already graded: {len(existing)}, New to grade: {len(to_grade)}")

    results = list(existing.values())
    grader = grader or VisionGrader()

    if to_grade:
        _run_grading(
            to_grade,
            lambda pdf: grader.grade_student_single_question(
                str(pdf), system_prompt, question_num
            ),
            results,
            lambda r: save_results(r, output_file, question_num=question_num),
        )

    save_results(results, output_file, question_num=question_num)
    _print_single_summary(results, question_num, grader)
    print(f"\nResults saved to: {output_file}")


def _print_single_summary(results: List[Dict[str, Any]], question_num: int, grader: VisionGrader):
    print(f"\n--- Question {question_num} Grading Summary ---")
    print(f"Total Students: {len(results)}")

    conflicts = [r for r in results if r.get("conflict", False)]
    print(f"Needs review (diff > {grader.conflict_limit} or a model failed): {len(conflicts)}")

    for c in conflicts[:10]:
        # .get() throughout: an entry may be an error result.
        g1 = c.get("model_1", {}).get("grade")
        g2 = c.get("model_2", {}).get("grade")
        print(f"  {c['student_id']}: {grader.model_1}={g1}, {grader.model_2}={g2}, "
              f"final={c.get('final_grade')}")
    if len(conflicts) > 10:
        print(f"  ... and {len(conflicts) - 10} more")

    grades = [r["final_grade"] for r in results if r.get("final_grade") is not None]
    if grades:
        print(f"\nAverage: {sum(grades) / len(grades):.1f}, "
              f"Highest: {max(grades)}, Lowest: {min(grades)}")


def grade_question_parts(
    question_num: int,
    part_labels: List[str],
    pdf_folder: str = PDF_FOLDER,
    answers_folder: str = ANSWERS_FOLDER,
    output_file: str = None,
    grader: VisionGrader = None,
    limit: int = 0,
):
    """Grade one question, part by part, for every student."""
    output_file = output_file or result_path(f"grades_question_{question_num}_parts.json")

    print(f"Loading correct answer for question {question_num}...")
    correct_answer = load_correct_answer(answers_folder, question_num)
    if correct_answer is None:
        print(f"Error: {answers_folder}/correct_answer_{question_num}.md not found!")
        return

    grader = grader or VisionGrader()
    # The grader checks evidence quotes against the marking scheme, so it needs
    # the text. Set here rather than at construction so callers cannot forget.
    grader.reference = correct_answer
    system_prompt = get_grading_prompt_parts(
        question_num, part_labels, correct_answer, grader.tag_scheme, grader.posture
    )

    pdf_files = _collect_pdfs(pdf_folder)
    if not pdf_files:
        return

    existing = load_existing_results(output_file)
    to_grade = [p for p in pdf_files if student_id_for(p) not in existing]
    if limit:
        # Deliberately bounded runs: measure cost on a sample before committing
        # the whole class.
        to_grade = to_grade[:limit]
    print(f"Already graded: {len(existing)}, New to grade: {len(to_grade)}")

    results = list(existing.values())

    save = lambda r: save_results(  # noqa: E731
        r, output_file, question_num=question_num,
        part_labels=part_labels, usage=grader.usage,
        tag_scheme=grader.tag_scheme, posture=grader.posture,
        samples=grader.samples,
    )

    if to_grade:
        _run_grading(
            to_grade,
            lambda pdf: grader.grade_student_question_parts(
                str(pdf), system_prompt, question_num, part_labels
            ),
            results,
            save,
        )

    save(results)
    _print_parts_summary(results, question_num, part_labels, grader)
    _print_cost_summary(grader.usage, len(to_grade))
    print(f"\nResults saved to: {output_file}")


def _result_tags(result: Dict[str, Any]) -> List[Optional[str]]:
    """
    The tag list from a result, ordered by part.

    Reads `part_constructs` as well: results written before tag schemes were
    configurable used that key for the SQL construct labels, and a resumed run
    should still print them rather than showing every old row as unreported.
    """
    tags = result.get("part_tags")
    if tags is None:
        tags = result.get("part_constructs")
    return list(tags.values()) if isinstance(tags, dict) else []


def _print_parts_summary(
    results: List[Dict[str, Any]], question_num: int,
    part_labels: List[str], grader: VisionGrader,
):
    print(f"\n--- Question {question_num} Grading Summary (by part) ---")
    print(f"Total Students: {len(results)}")

    for i, label in enumerate(part_labels):
        grades = [
            r["final_grades"][i]
            for r in results
            if i < len(r.get("final_grades", [])) and r["final_grades"][i] is not None
        ]
        flagged = sum(
            1 for r in results
            if i < len(r.get("conflicts", [])) and r["conflicts"][i]
        )
        if grades:
            print(f"  Part ({label}): avg {sum(grades) / len(grades):5.1f}  "
                  f"min {min(grades):3}  max {max(grades):3}  "
                  f"n={len(grades)}  flagged={flagged}")
        else:
            print(f"  Part ({label}): no grades  flagged={flagged}")

    scheme = grader.tag_scheme

    # Tag tallies, per part. Descriptive: these do not affect grades.
    if scheme:
        print(f"\n{scheme.title} (does not affect grades):")
        for i, label in enumerate(part_labels):
            tally: Dict[str, int] = {}
            for r in results:
                c = _result_tags(r)
                v = c[i] if i < len(c) else None
                tally[v or "unreported"] = tally.get(v or "unreported", 0) + 1
            summary = "  ".join(f"{k}={v}" for k, v in sorted(tally.items()))
            print(f"  Part ({label}): {summary}")

    # Short markers so a whole class fits on screen.
    marks = scheme.markers if scheme else {}
    legend = (
        ";  marker: "
        + " ".join(f"{lb.marker}={lb.name}" for lb in scheme.labels)
        + " ?=unreported"
    ) if scheme else ""
    print(f"\nPer-student grades (parts {', '.join(part_labels)}){legend}")
    for r in sorted(results, key=lambda x: x["student_id"]):
        cells = []
        tags = _result_tags(r)
        for i, label in enumerate(part_labels):
            g = r["final_grades"][i] if i < len(r.get("final_grades", [])) else None
            flag = "*" if i < len(r.get("conflicts", [])) and r["conflicts"][i] else " "
            mark = marks.get(tags[i] if i < len(tags) else None, "?") if scheme else ""
            # "null" and 0 must stay visibly different: 0 is a real zero,
            # null is "we have no grade".
            cells.append(f"{'----' if g is None else f'{g:4}'}{mark}{flag}")
        print(f"  {r['student_id']:34} {' '.join(cells)}")

    _print_assurance_summary(results, part_labels, grader)

    needs_review = [r for r in results if r.get("has_conflict", False)]
    if grader.single:
        print(f"\nNeeds review (* = the model produced no grade for that part): "
              f"{len(needs_review)} students")
        print(f"Single-model run ({grader.model_1}): grades are not cross-checked.")
    else:
        print(f"\nNeeds review (* = models differed by > {grader.conflict_limit}, "
              f"or a model produced no grade): {len(needs_review)} students")


def _result_assurance(result: Dict[str, Any], part_labels: List[str]) -> List[Dict[str, Any]]:
    """The per-part assurance records from a result, ordered by part.

    Returns empty dicts for results written before assurance existed, so a
    resumed run prints old rows as "not reported" rather than crashing.
    """
    a = result.get("assurance")
    if not isinstance(a, dict):
        return [{} for _ in part_labels]
    return [a.get(p) or {} for p in part_labels]


def _print_assurance_summary(
    results: List[Dict[str, Any]], part_labels: List[str], grader: VisionGrader
):
    """
    Report what was withheld and why, and how much the model agreed with itself.

    The withheld counts matter more than they look. Every one of them is a
    student who would previously have received a silent 0 for a part that was
    unreadable or missing from the file -- a mark against work that may well
    exist, and the failure mode this system is least able to notice on its own.
    """
    withheld: Dict[str, int] = {}
    for r in results:
        for a in _result_assurance(r, part_labels):
            if a.get("veto"):
                withheld[a["veto"]] = withheld.get(a["veto"], 0) + 1

    print("\nGrades withheld (not scored 0 -- these need a human):")
    if withheld:
        explain = {
            "not_found": "no answer found in the file (a missing page?)",
            "illegible": "could not be read",
            "unattempted_null": "blank, recorded as null by posture",
            "unreadable_pdf": "the PDF itself could not be opened",
        }
        for veto, count in sorted(withheld.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4} part(s): {explain.get(veto, veto)}")
    else:
        print("  none")

    ungrounded = [
        (r["student_id"], part_labels[i])
        for r in results
        for i, a in enumerate(_result_assurance(r, part_labels))
        if a.get("evidence_ok") is False
    ]
    if ungrounded:
        print(f"\nGrades with no usable evidence quote: {len(ungrounded)}")
        for sid, part in ungrounded[:10]:
            print(f"  {sid} part ({part})")
        if len(ungrounded) > 10:
            print(f"  ... and {len(ungrounded) - 10} more")
        print("  A quote that is missing, or that came from the marking scheme "
              "rather than\n  the submission, is the signature of a model that "
              "is not reading the pages.")

    if grader.samples <= 1:
        return

    print(f"\nSelf-agreement across {grader.samples} samples of {grader.model_1}:")
    for i, label in enumerate(part_labels):
        tally: Dict[str, int] = {}
        spreads: List[int] = []
        for r in results:
            a = _result_assurance(r, part_labels)[i]
            level = a.get("confidence")
            if level:
                tally[level] = tally.get(level, 0) + 1
            if a.get("spread") is not None:
                spreads.append(a["spread"])
        if not tally:
            continue
        summary = "  ".join(f"{k}={v}" for k, v in sorted(tally.items()))
        worst = f"  worst spread {max(spreads)}" if spreads else ""
        mean = f"  mean spread {sum(spreads) / len(spreads):.1f}" if spreads else ""
        print(f"  Part ({label}): {summary}{mean}{worst}")

    print(f"\n  'high' means the samples fell within {grader.conflict_limit} marks "
          f"of each other --\n  the same limit used to decide two different models "
          f"disagree. A part that is\n  not high here is a part whose grade would "
          f"not reproduce on a re-run.")


def _print_cost_summary(usage: UsageLedger, students_graded: int):
    totals = usage.by_model()
    if not totals:
        return

    print("\n--- Cost ---")
    print(f"{'model':40}{'calls':>7}{'failed':>8}{'prompt tok':>12}"
          f"{'out tok':>9}{'cost $':>10}")
    for model, t in sorted(totals.items(), key=lambda kv: -kv[1]["cost"]):
        print(f"{model:40}{int(t['calls']):7}{int(t['failed']):8}"
              f"{int(t['prompt_tokens']):12}{int(t['completion_tokens']):9}"
              f"{t['cost']:10.5f}")

    total = usage.total_cost
    print(f"{'TOTAL':40}{'':7}{'':8}{'':12}{'':9}{total:10.5f}")
    if students_graded:
        per = total / students_graded
        print(f"\n${per:.5f} per student ({students_graded} graded this run)")
        print(f"Extrapolated: ${per * 100:.2f} per 100 students, "
              f"${per * 130:.2f} per 130-student class")
    if total == 0:
        print("\nNote: the gateway reported no cost figures. Token counts above "
              "are still real; only the dollar amounts are missing.")


def grade_all_questions(
    num_questions: int,
    pdf_folder: str = PDF_FOLDER,
    answers_folder: str = ANSWERS_FOLDER,
    output_file: str = OUTPUT_FILE,
    grader: VisionGrader = None,
):
    """Grade every question for every student in a single pass per model."""
    print(f"Loading correct answers for {num_questions} questions...")
    correct_answers = load_all_correct_answers(answers_folder, num_questions)
    system_prompt = get_grading_prompt_all(num_questions, correct_answers)

    pdf_files = _collect_pdfs(pdf_folder)
    if not pdf_files:
        return

    existing = load_existing_results(output_file)
    to_grade = [p for p in pdf_files if student_id_for(p) not in existing]
    print(f"Already graded: {len(existing)}, New to grade: {len(to_grade)}")

    results = list(existing.values())
    grader = grader or VisionGrader()

    if to_grade:
        _run_grading(
            to_grade,
            lambda pdf: grader.grade_student(str(pdf), system_prompt, num_questions),
            results,
            lambda r: save_results(r, output_file, num_questions=num_questions),
        )

    save_results(results, output_file, num_questions=num_questions)
    _print_all_summary(results, num_questions)
    print(f"\nResults saved to: {output_file}")


def _print_all_summary(results: List[Dict[str, Any]], num_questions: int):
    print("\n--- Grading Summary ---")
    print(f"Total Students: {len(results)}")

    needs_review = [r for r in results if r.get("has_conflict", False)]
    print(f"Students needing review: {len(needs_review)}")

    for s in needs_review[:10]:
        flagged = [i + 1 for i, c in enumerate(s.get("conflicts", [])) if c]
        print(f"  {s['student_id']}: questions {flagged}")
    if len(needs_review) > 10:
        print(f"  ... and {len(needs_review) - 10} more")

    print("\nAverage grade per question:")
    for q in range(num_questions):
        # Guard the index: a resumed run may hold results from a previous run
        # with a different question count.
        grades = [
            r["final_grades"][q]
            for r in results
            if q < len(r.get("final_grades", [])) and r["final_grades"][q] is not None
        ]
        if grades:
            print(f"  Q{q + 1}: {sum(grades) / len(grades):.1f}  (n={len(grades)})")
        else:
            print(f"  Q{q + 1}: no grades")


def main():
    parser = argparse.ArgumentParser(description="Vision-based homework grader")
    parser.add_argument("--question", "-q", type=int,
                        help="Grade only this question number")
    parser.add_argument("--num-questions", "-n", type=int,
                        help="Total number of questions (grade all at once)")
    parser.add_argument("--parts", default=QUESTION_PARTS or None,
                        help="Comma-separated part labels to grade separately "
                             "within --question, e.g. 'a,b,c'. Omit to grade "
                             "the question as a whole.")
    parser.add_argument("--tags", default=TAG_SCHEME or None,
                        help="Tag scheme recorded alongside each grade: a name "
                             f"from {TAGSETS_FOLDER}/ ({', '.join(sorted(list_tag_schemes())) or 'none'}) "
                             "or a path to a JSON scheme. Tags never affect grades.")
    parser.add_argument("--posture", default=POSTURE or None,
                        help="How to make the marginal calls: a name from "
                             f"{POSTURES_FOLDER}/ ({', '.join(sorted(list_postures())) or 'none'}) "
                             "or a path to a JSON posture. Affects grades, and "
                             "is recorded in full in the results.")
    parser.add_argument("--samples", type=int, default=GRADE_SAMPLES,
                        help="Grade each submission this many times with "
                             "--model-1 and report the spread as a confidence. "
                             "1 (default) disables sampling.")
    parser.add_argument("--limit", "-l", type=int, default=0,
                        help="Grade at most this many new students (0 = all)")
    parser.add_argument("--single", action="store_true",
                        help="Grade with --model-1 only: half the cost, no "
                             "cross-checking between models")
    parser.add_argument("--pdf-folder", "-p", default=PDF_FOLDER,
                        help="Folder containing student PDFs")
    parser.add_argument("--answers-folder", "-a", default=ANSWERS_FOLDER,
                        help="Folder containing correct answers")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--model-1", default=MODEL_1_NAME, help="First grading model")
    parser.add_argument("--model-2", default=MODEL_2_NAME, help="Second grading model")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help="Render resolution for PDF pages")
    parser.add_argument("--conflict-limit", type=int, default=GRADE_CONFLICT_LIMIT,
                        help="Grade difference above which to flag for review")
    parser.add_argument("--resolution", choices=["max", "mean", "min"],
                        default=GRADE_RESOLUTION,
                        help="How to combine two agreeing grades")
    args = parser.parse_args()

    if not args.question and not args.num_questions:
        parser.error(
            "specify either --question N (one question) "
            "or --num-questions N (all questions)"
        )

    if args.parts and not args.question:
        parser.error("--parts applies to a single question; add --question N")

    # Parts named for this question in course.toml apply unless --parts says otherwise.
    if not args.parts and args.question and question_parts(args.question):
        args.parts = ",".join(question_parts(args.question))

    part_labels = [p.strip() for p in args.parts.split(",") if p.strip()] if args.parts else []

    try:
        tag_scheme = load_tag_scheme(args.tags)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if tag_scheme and not part_labels:
        # Tagging is wired into the per-part path only. Failing loudly beats
        # running the whole class and silently producing no tags at all.
        parser.error("--tags currently applies to per-part grading; add --parts")

    try:
        posture = load_posture(args.posture)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.samples < 1:
        parser.error("--samples must be at least 1")

    if args.samples > 1 and not part_labels:
        # Same reasoning as --tags below: sampling is wired into the per-part
        # path only, and failing now beats discovering it after a whole class.
        parser.error("--samples currently applies to per-part grading; add --parts")

    if tag_scheme:
        print(f"Tagging with '{tag_scheme.key}' "
              f"({', '.join(tag_scheme.names)}) -- does not affect grades.")

    # Posture is always printed, including the default. It changes grades, so a
    # run whose banner does not say how it marked is a run whose output cannot
    # be explained later.
    print(f"Posture '{posture.key}': {posture.summary()}")

    if args.samples > 1:
        print(f"Sampling: grading each submission {args.samples}x with "
              f"{args.model_1} to measure self-agreement "
              f"(~{args.samples}x the cost of a single-model run).")

    if args.single:
        print(f"Single-model run: grading with {args.model_1} only. "
              f"Grades are not cross-checked.")
    elif args.model_1 == args.model_2:
        print(
            f"Warning: both graders are set to '{args.model_1}'. Dual-model "
            f"verification only detects disagreement between different models."
        )

    grader = VisionGrader(
        model_1=args.model_1,
        model_2=args.model_2,
        dpi=args.dpi,
        conflict_limit=args.conflict_limit,
        resolution=args.resolution,
        single=args.single,
        tag_scheme=tag_scheme,
        posture=posture,
        samples=args.samples,
    )

    if part_labels:
        grade_question_parts(
            question_num=args.question,
            part_labels=part_labels,
            pdf_folder=args.pdf_folder,
            answers_folder=args.answers_folder,
            output_file=args.output,
            grader=grader,
            limit=args.limit,
        )
    elif args.question:
        grade_single_question(
            question_num=args.question,
            pdf_folder=args.pdf_folder,
            answers_folder=args.answers_folder,
            output_file=args.output,
            grader=grader,
        )
    else:
        grade_all_questions(
            num_questions=args.num_questions,
            pdf_folder=args.pdf_folder,
            answers_folder=args.answers_folder,
            output_file=args.output or OUTPUT_FILE,
            grader=grader,
        )


if __name__ == "__main__":
    main()
