"""
Parsing model output into grades, and reconciling two models' grades.

The parsing here is deliberately strict.  An earlier version fell back to
"take the last number between 0 and 100 anywhere in the response", which
happily turned a sentence like "the student used 6 transistors" into a grade
of 6.  A missing grade must stay missing so it can be flagged, never guessed.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .config import GRADE_CONFLICT_LIMIT, GRADE_RESOLUTION

# Matches "GRADE: 85", "GRADE: [85]", "grade = 85", "**GRADE:** 85", "GRADE: 85/100"
_SINGLE_GRADE_RE = re.compile(
    r"GRADE\s*\**\s*[:=]\s*\**\s*\[?\s*(\d{1,3})\s*\]?",
    re.IGNORECASE,
)

# Matches "GRADES: [85, 70, 100]" and tolerates "null"/"-" for unanswered items.
_GRADE_ARRAY_RE = re.compile(
    r"GRADES\s*\**\s*[:=]\s*\**\s*\[([^\]]*)\]",
    re.IGNORECASE,
)

# Matches a tag array such as "CONSTRUCTS: [none, cte, cte+subquery]" -- a
# per-item note about the answer, e.g. which SQL constructs it used. Purely
# descriptive: it is recorded alongside the grade and must never change it.
# The field name is supplied by the tag scheme, so the vocabulary is not
# hardcoded here; see tagging.py.
_TAG_ARRAY_TEMPLATE = r"{field}\s*\**\s*[:=]\s*\**\s*\[([^\]]*)\]"

# Matches a quoted string inside an array, tolerating both straight and curly
# quotes: models copying Persian handwriting emit typographic quotes often
# enough that requiring ASCII ones loses real evidence.
_QUOTED_RE = re.compile(r'"([^"]*)"|“([^”]*)”|\'([^\']*)\'')


def _clamp(value: int) -> int:
    return min(100, max(0, value))


def extract_single_grade(text: str) -> Optional[int]:
    """
    Pull a single 0-100 grade out of a model response.

    Returns None if the model did not state a grade in the required format.
    Callers must treat None as "grading failed", not as zero.
    """
    if not text:
        return None

    # Use the last match: models often restate the grade at the end after
    # revising their reasoning, and the final word is the one that counts.
    matches = _SINGLE_GRADE_RE.findall(text)
    if matches:
        return _clamp(int(matches[-1]))

    return None


def extract_grades(text: str, num_questions: int) -> List[Optional[int]]:
    """
    Pull an array of grades out of a model response.

    Always returns exactly num_questions entries.  Entries the model did not
    provide (or provided as null) come back as None.
    """
    empty: List[Optional[int]] = [None] * num_questions
    if not text:
        return empty

    matches = _GRADE_ARRAY_RE.findall(text)
    if not matches:
        return empty

    grades: List[Optional[int]] = []
    for token in matches[-1].split(","):
        token = token.strip().strip("\"'")
        if not token or token.lower() in {"null", "none", "-", "n/a", "na"}:
            grades.append(None)
            continue
        try:
            grades.append(_clamp(int(float(token))))
        except ValueError:
            # A single unparseable entry should not discard the whole array.
            grades.append(None)

    # Normalise length: pad short arrays, drop anything past num_questions.
    if len(grades) < num_questions:
        grades.extend([None] * (num_questions - len(grades)))
    return grades[:num_questions]


def extract_tags(
    text: str,
    num_items: int,
    field: str,
    allowed: Sequence[str],
    aliases: Optional[Dict[str, str]] = None,
) -> List[Optional[str]]:
    """
    Pull an array of descriptive labels out of a response.

    `field` is the output line the model was told to write (e.g. "CONSTRUCTS"),
    `allowed` the scheme's vocabulary. Always returns exactly num_items entries.
    An item the model did not label comes back as None, meaning "not reported"
    -- distinct from a label like "none", which is a positive statement that
    the grader looked and found nothing.

    Labels outside the vocabulary are discarded rather than stored, so
    downstream filters can rely on the scheme's vocabulary being exhaustive.
    """
    empty: List[Optional[str]] = [None] * num_items
    if not text or not field:
        return empty

    pattern = re.compile(
        _TAG_ARRAY_TEMPLATE.format(field=re.escape(field)), re.IGNORECASE
    )
    matches = pattern.findall(text)
    if not matches:
        return empty

    allowed_set = {a.lower() for a in allowed}
    alias_map = {k.lower(): v.lower() for k, v in (aliases or {}).items()}

    labels: List[Optional[str]] = []
    for token in matches[-1].split(","):
        token = token.strip().strip("\"'`").lower().replace(" ", "")
        token = alias_map.get(token, token)
        labels.append(token if token in allowed_set else None)

    if len(labels) < num_items:
        labels.extend([None] * (num_items - len(labels)))
    return labels[:num_items]


def extract_evidence(text: str, num_items: int, field: str = "EVIDENCE") -> List[str]:
    """
    Pull an array of verbatim quotes out of a response.

    Quotes contain commas -- a SQL line almost always does -- so this cannot use
    the comma splitting the tag and grade arrays rely on. It matches quoted
    strings instead, which also means an unquoted entry is skipped rather than
    silently mangling the ones around it.

    Always returns exactly num_items entries; a part the model did not quote
    comes back as "". An empty quote is not an error on its own -- an
    unattempted part has nothing to quote -- so callers pair this with the
    grade before deciding anything (see assurance.check_evidence).
    """
    empty = [""] * num_items
    if not text or not field:
        return empty

    pattern = re.compile(
        _TAG_ARRAY_TEMPLATE.format(field=re.escape(field)), re.IGNORECASE
    )
    matches = pattern.findall(text)
    if not matches:
        return empty

    quotes = [
        next(g for g in groups if g is not None)
        for groups in _QUOTED_RE.findall(matches[-1])
    ]

    if len(quotes) < num_items:
        quotes.extend([""] * (num_items - len(quotes)))
    return quotes[:num_items]


def resolve_grade(
    grade1: Optional[int],
    grade2: Optional[int],
    conflict_limit: int = GRADE_CONFLICT_LIMIT,
    resolution: str = GRADE_RESOLUTION,
) -> Tuple[Optional[int], bool]:
    """
    Combine two models' grades for one question.

    Returns (final_grade, needs_review).

    needs_review is True whenever a human should look at this: the models
    disagreed beyond the limit, or at least one of them failed to produce a
    grade at all.  A single model's opinion is never silently accepted as
    though both had agreed.
    """
    if grade1 is None and grade2 is None:
        # Nothing to go on. Leave the grade unset rather than writing a 0 that
        # looks like a real "the student got everything wrong" score.
        return None, True

    if grade1 is None or grade2 is None:
        # One model failed. Use the surviving grade, but flag it: it has not
        # been cross-checked, which is the entire point of dual-model grading.
        return (grade1 if grade1 is not None else grade2), True

    if resolution == "mean":
        final = round((grade1 + grade2) / 2)
    elif resolution == "min":
        final = min(grade1, grade2)
    else:
        final = max(grade1, grade2)

    return final, abs(grade1 - grade2) > conflict_limit


def resolve_grades(
    grades1: List[Optional[int]],
    grades2: List[Optional[int]],
    conflict_limit: int = GRADE_CONFLICT_LIMIT,
    resolution: str = GRADE_RESOLUTION,
) -> Tuple[List[Optional[int]], List[bool]]:
    """Apply resolve_grade across a whole question array."""
    finals: List[Optional[int]] = []
    conflicts: List[bool] = []

    length = max(len(grades1), len(grades2))
    for i in range(length):
        g1 = grades1[i] if i < len(grades1) else None
        g2 = grades2[i] if i < len(grades2) else None
        final, conflict = resolve_grade(g1, g2, conflict_limit, resolution)
        finals.append(final)
        conflicts.append(conflict)

    return finals, conflicts
