"""
Knowing how much to trust a grade.

Three separate mechanisms live here, and they answer three different questions:

**Can the model read it, and is it even there?**  `legibility` and `presence`
are reported per part and can *veto* a grade.  This exists because the worst
failure this system has is silent and one-directional: a model that cannot read
Persian handwriting returns 0, and a part whose page failed to scan is graded 0,
and both look exactly like a student who wrote nothing.  A zero is a claim about
the student's work; it may only be written when there is work to make a claim
about.  See ADR-003 -- this is the same `null` vs `0` rule, pushed one step
back to its cause.

**How much does the model agree with itself?**  Grade the same submission N
times and the spread across those samples is the confidence.  This is measured,
not self-reported: asking a model "how confident are you, 0-100" produces a
number in the high eighties almost regardless of whether it is right, and an
uncalibrated confidence is worse than none because it gets trusted.  The
motivating observation is real -- one HW3 submission scored 40 and then 20 on a
re-run of the same model at temperature 0, twice the conflict limit used to
decide two *different* models disagree.

**Is the model reading the page, or reciting the answer key?**  Evidence quotes
are required per part, and a quote that turns out to be lifted from the marking
scheme rather than from the student is the signature of confabulation -- that is
precisely what gpt-4.1-nano did when it asserted "no subqueries or CTEs" about
pages beginning `WITH E AS (...)`.  Catching that took a human rendering page 4
to PNG by hand; `evidence_echoes_reference` does it on every grade for free.

Note the asymmetry throughout: every mechanism here can withhold or flag a
grade, and none can raise one.  A trust signal that could increase a mark would
be a grading criterion, and grading criteria belong in the marking scheme.
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- Vocabularies -------------------------------------------------------------

# Can the model make out what is written?
LEGIBILITY_FIELD = "LEGIBILITY"
LEGIBILITY_VALUES = ("clear", "partial", "illegible")
LEGIBILITY_ALIASES = {
    "legible": "clear",
    "readable": "clear",
    "unclear": "partial",
    "partially_legible": "partial",
    "partly": "partial",
    "unreadable": "illegible",
    "illegable": "illegible",  # models misspell this one constantly
}

# Is there an answer to grade at all?
PRESENCE_FIELD = "PRESENCE"
PRESENCE_VALUES = ("answered", "unattempted", "not_found")
PRESENCE_ALIASES = {
    "present": "answered",
    "answered_partially": "answered",
    "attempted": "answered",
    "blank": "unattempted",
    "empty": "unattempted",
    "no_attempt": "unattempted",
    "missing": "not_found",
    "notfound": "not_found",
    "absent": "not_found",
}

EVIDENCE_FIELD = "EVIDENCE"

# A quote shorter than this is too generic to tell us anything -- "SELECT" is
# in every submission and every marking scheme alike.
MIN_EVIDENCE_CHARS = 12


# --- The prompt block ---------------------------------------------------------


def prompt_section(item_word: str = "part") -> str:
    """The system-prompt block defining legibility, presence and evidence."""
    return f"""### Readability and evidence (reported separately from the grade)

For each {item_word}, report three things alongside the grade.

**{LEGIBILITY_FIELD}** -- can you actually make out what the student wrote?
- `clear` -- you can read it
- `partial` -- you can read some of it, but parts are genuinely unclear
- `illegible` -- you cannot read it well enough to judge it

**{PRESENCE_FIELD}** -- is there an answer to this {item_word} at all?
- `answered` -- the student attempted it, correctly or not
- `unattempted` -- the student left it blank, or wrote something that is plainly
  not an attempt ("I ran out of time", a doodle, a request for marks)
- `not_found` -- there is no answer to this {item_word} anywhere in these pages.
  Use this when the {item_word} appears to be missing altogether, for instance
  if a page seems absent from the scan. It is NOT the same as `unattempted`:
  `unattempted` means you found the place where the answer should be and it was
  empty; `not_found` means you never found that place.

**{EVIDENCE_FIELD}** -- quote, VERBATIM, the single most important line the
student wrote for this {item_word}, the one your grade turns on. Copy it exactly
as it appears on the page. If the {item_word} is unattempted or not found, write
an empty string.

Three rules about these fields:

1. If you cannot read a {item_word}, mark it `illegible`. Do NOT guess a grade
   and do NOT give it 0. "I could not read this" and "the student got this
   wrong" are completely different findings and must not be reported the same
   way. A grade of 0 on work you could not read would be recorded against a
   student who may have answered perfectly.
2. If a {item_word} is not present in these pages, mark it `not_found`, for the
   same reason. A missing page is not a wrong answer.
3. The {EVIDENCE_FIELD} quote must come from the STUDENT'S submission. Never
   quote the reference solution or the marking scheme back. If you cannot find a
   line in the student's own writing to quote, that itself means you have not
   located their answer -- say so."""


def output_spec(count: int, ordering: str) -> str:
    """The lines describing these arrays in the required output format."""
    leg = ", ".join((["clear"] * count)[:count])
    pres = ", ".join((["answered"] * count)[:count])
    ev = ", ".join(f'"<verbatim quote {i + 1}>"' for i in range(count))
    return (
        f"{LEGIBILITY_FIELD}: [{leg}]\n"
        f"{PRESENCE_FIELD}: [{pres}]\n"
        f"{EVIDENCE_FIELD}: [{ev}]\n"
        f"\nEach of these arrays MUST contain exactly {count} entries, in "
        f"{ordering}. {LEGIBILITY_FIELD} uses only "
        f"{', '.join(f'`{v}`' for v in LEGIBILITY_VALUES)}; {PRESENCE_FIELD} "
        f"uses only {', '.join(f'`{v}`' for v in PRESENCE_VALUES)}; "
        f"{EVIDENCE_FIELD} entries are double-quoted strings copied from the "
        f"student's page."
    )


# --- Grade vetoes -------------------------------------------------------------

# Why a grade was withheld. Recorded per part so a reviewer knows which of the
# three quite different problems they are looking at.
VETO_NOT_FOUND = "not_found"
VETO_ILLEGIBLE = "illegible"
VETO_UNATTEMPTED = "unattempted_null"


@dataclass
class PartAssurance:
    """Everything known about how much to trust one part's grade."""

    legibility: Optional[str] = None
    presence: Optional[str] = None
    evidence: str = ""
    evidence_ok: Optional[bool] = None   # None = not checked
    veto: Optional[str] = None           # why the grade was withheld
    confidence: Optional[str] = None     # high | medium | low | unmeasured
    spread: Optional[int] = None
    samples: List[Optional[int]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "legibility": self.legibility,
            "presence": self.presence,
            "evidence": self.evidence,
        }
        if self.evidence_ok is not None:
            d["evidence_ok"] = self.evidence_ok
        if self.veto:
            d["veto"] = self.veto
        if self.confidence:
            d["confidence"] = self.confidence
            d["spread"] = self.spread
            d["samples"] = self.samples
        return d


def apply_vetoes(
    grades: Sequence[Optional[int]],
    legibility: Sequence[Optional[str]],
    presence: Sequence[Optional[str]],
    unattempted: str = "zero",
) -> Tuple[List[Optional[int]], List[bool], List[Optional[str]]]:
    """
    Withhold grades the model was not entitled to give.

    Returns (grades, flags, vetoes) with one entry per part.

    The rules, in order of precedence:

    - `not_found` -> no grade. The part is not in the pages, so nothing about
      the student's work has been observed. Grading this 0 is the missing-page
      failure: a mark of zero recorded against work that may exist on a page
      that did not scan.
    - `illegible` -> no grade, for the same reason at a different cause. Several
      cheap models return 0 for Persian handwriting they cannot read; this makes
      that admission mandatory instead of optional, and makes it visible.
    - `unattempted` -> 0 by default, because a blank answer genuinely is worth
      nothing. Courses that want the distinction preserved for a later manual
      decision set the posture's `unattempted` to "null".

    Nothing here ever raises a grade or creates one where the model gave none.
    """
    out_grades: List[Optional[int]] = []
    flags: List[bool] = []
    vetoes: List[Optional[str]] = []

    for i, grade in enumerate(grades):
        leg = legibility[i] if i < len(legibility) else None
        pres = presence[i] if i < len(presence) else None
        veto: Optional[str] = None
        result = grade

        if pres == "not_found":
            result, veto = None, VETO_NOT_FOUND
        elif leg == "illegible":
            result, veto = None, VETO_ILLEGIBLE
        elif pres == "unattempted" and unattempted == "null":
            result, veto = None, VETO_UNATTEMPTED

        out_grades.append(result)
        vetoes.append(veto)
        # A vetoed part always needs a human: someone has to decide whether to
        # re-scan, grade it by hand, or record a zero deliberately.
        flags.append(veto is not None)

    return out_grades, flags, vetoes


# --- Confidence from repeated sampling ----------------------------------------

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_UNMEASURED = "unmeasured"


def confidence_level(spread: Optional[int], conflict_limit: int) -> str:
    """
    Turn a numeric spread into a label.

    The thresholds are pinned to `conflict_limit` on purpose: that is already
    the project's stated view of how far apart two gradings can be before a
    human should look, so the same number should govern a model disagreeing
    with itself. If sampling shows intra-model spread routinely exceeding it,
    that is a finding about the limit, not a reason to widen it here.
    """
    if spread is None:
        return CONFIDENCE_UNMEASURED
    if spread <= conflict_limit:
        return CONFIDENCE_HIGH
    if spread <= 2 * conflict_limit:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def summarise_samples(
    samples: Sequence[Optional[int]], conflict_limit: int
) -> Tuple[Optional[int], Optional[int], str]:
    """
    Reduce N gradings of one part to (grade, spread, confidence).

    The median is used rather than the mean or the max. The max -- which is the
    project's default `GRADE_RESOLUTION` for two agreeing models -- is actively
    wrong across repeated samples of one model: taking the maximum of N draws
    from a noisy distribution is a systematic upward bias that grows with N,
    not an estimate of anything. The median is robust to a single wild sample,
    which is the failure mode actually observed.

    Samples where the model produced no grade are excluded from the median but
    counted in the spread decision: if a model graded a part three times out of
    five, that is not a confident grade.
    """
    values = [s for s in samples if s is not None]
    if not values:
        return None, None, CONFIDENCE_UNMEASURED

    grade = int(round(statistics.median(values)))

    if len(samples) < 2:
        return grade, None, CONFIDENCE_UNMEASURED

    spread = max(values) - min(values)
    level = confidence_level(spread, conflict_limit)

    # A part some samples failed to grade at all is not high-confidence, however
    # tightly the surviving samples agree.
    if len(values) < len(samples) and level == CONFIDENCE_HIGH:
        level = CONFIDENCE_MEDIUM

    return grade, spread, level


def summarise_all_samples(
    sample_runs: Sequence[Sequence[Optional[int]]],
    num_parts: int,
    conflict_limit: int,
) -> Tuple[List[Optional[int]], List[Optional[int]], List[str], List[List[Optional[int]]]]:
    """
    Apply `summarise_samples` across every part.

    `sample_runs` is one list of per-part grades per sample; the return is
    per-part (grades, spreads, levels, samples).
    """
    per_part: List[List[Optional[int]]] = [
        [run[i] if i < len(run) else None for run in sample_runs]
        for i in range(num_parts)
    ]
    grades, spreads, levels = [], [], []
    for values in per_part:
        g, s, lvl = summarise_samples(values, conflict_limit)
        grades.append(g)
        spreads.append(s)
        levels.append(lvl)
    return grades, spreads, levels, per_part


# --- Evidence checking --------------------------------------------------------


def _normalise(text: str) -> str:
    """Collapse whitespace and case so quoting differences do not matter."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def evidence_echoes_reference(quote: str, reference: str) -> bool:
    """
    True when an "evidence" quote appears to come from the marking scheme.

    This is the confabulation test. A model that is genuinely reading the pages
    quotes the student; a model that is paraphrasing the answer key back quotes
    the answer key. The latter produced grades that looked entirely plausible
    and were invented, and cost a 20-student trial to detect by hand.

    Short quotes are exempt: a fragment like "SELECT u, v" legitimately appears
    in the student's work and the reference alike, and flagging those would
    bury the real signal in noise.
    """
    q = _normalise(quote)
    if len(q) < MIN_EVIDENCE_CHARS:
        return False
    return q in _normalise(reference)


def check_evidence(
    quotes: Sequence[str],
    grades: Sequence[Optional[int]],
    reference: str,
) -> Tuple[List[Optional[bool]], List[str]]:
    """
    Check each part's evidence quote. Returns (ok_flags, problems).

    `None` in ok_flags means "not checked" -- there is no grade to justify, so
    a missing quote is not a problem. A part with a grade and no usable quote
    is reported, because an ungrounded grade is exactly what the evidence
    requirement exists to surface.
    """
    ok: List[Optional[bool]] = []
    problems: List[str] = []

    for i, grade in enumerate(grades):
        quote = quotes[i] if i < len(quotes) else ""
        if grade is None:
            ok.append(None)
            continue

        if len(_normalise(quote)) < MIN_EVIDENCE_CHARS:
            ok.append(False)
            problems.append(f"part {i + 1}: graded {grade} with no evidence quote")
        elif evidence_echoes_reference(quote, reference):
            ok.append(False)
            problems.append(
                f"part {i + 1}: evidence quote appears in the marking scheme, "
                f"not the submission -- {quote[:60]!r}"
            )
        else:
            ok.append(True)

    return ok, problems
