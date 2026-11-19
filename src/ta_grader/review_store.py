"""
The review side of the system: turning a grades file into a queue of decisions.

Everything upstream of this module is about producing grades. This module is
about the scarce resource on the other side -- a TA's attention -- and it makes
three claims that shape the whole design.

**A results file is not a review queue.**  The grader emits one row per student
holding several grades. The unit a human actually reviews is one *part* of one
student: you look at part (c) on page 4, decide, and move on. So the file is
flattened into `ReviewItem`s, one per gradeable thing, and everything downstream
-- ordering, statistics, sign-off -- works on those.

**Attention is allocated, not spent uniformly.**  Reviewing in student-id order
means the 81st submission is reviewed with less care than the 3rd, which is the
exact failure the whole project exists to remove (see README). Items are ranked
by `uncertainty x marks_at_stake`, so the first thing a TA sees is the grade most
likely to be wrong by the largest number of marks. The ordering is a heuristic
and is documented as one; it does not need to be calibrated to beat filename
order, which is calibrated to nothing.

**Every override is evidence and must be kept.**  The old reviewer overwrote
`final_grade` in place, which destroyed the model's original judgement and with
it any chance of measuring accuracy later. Decisions are appended to a JSONL
ledger holding before, after, reason and timestamp. That ledger is a
hand-checked calibration set, accumulating for free over a term, and the
input to confidence calibration.

Three invariants from the grading side are load-bearing here too, because a
review tool can violate them just as easily as a grader can:

- `null` is not `0` (ADR-003). Accepting a suggestion for a part that neither
  model graded must not silently write a zero. The old `pick_higher` returned 0
  when both grades were None, which converted "we have no grade" into "the
  student got nothing right" on a single click.
- Results are written atomically (ADR-010). The reviewer writes the same file a
  background grader run may be appending to; a plain truncating write can lose a
  whole class.
- A single model's opinion is never presented as an agreed result (ADR-001, ADR-005).
  A `single_model` run has nothing to cross-check, and the queue says so rather
  than displaying a second grader's empty column as if it had agreed.
"""

import csv
import difflib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- Result-file shapes -------------------------------------------------------

FORMAT_PARTS = "parts"          # one question, graded part by part
FORMAT_SINGLE = "single"        # one question, graded whole
FORMAT_MULTI = "multi"          # several questions in one run


def detect_format(data: Dict[str, Any]) -> str:
    """
    Which of the three output shapes this file is.

    Ordering matters: a per-part file also carries `question`, so parts must be
    tested first. The previous reviewer tested `question`/`num_questions` only
    and therefore rendered the per-part HW3 file -- the only real production run
    this project has ever done -- as "Q1 Q2 Q3", with the part labels lost.
    """
    if data.get("parts"):
        return FORMAT_PARTS
    if "num_questions" in data:
        return FORMAT_MULTI
    return FORMAT_SINGLE


# --- Reasons an item needs a human --------------------------------------------

# Ordered by how much they should dominate the queue. The first four are not
# "this grade might be off"; they are "this is not a grade yet".
FLAG_NO_GRADE = "no_grade"
FLAG_VETO = "veto"
FLAG_EVIDENCE = "evidence"
FLAG_DISAGREEMENT = "disagreement"
FLAG_SPREAD = "spread"
FLAG_SINGLE_MODEL = "single_model"
FLAG_EXTREME = "extreme"

FLAG_TEXT = {
    FLAG_NO_GRADE: "no grade was produced",
    FLAG_VETO: "grade withheld",
    FLAG_EVIDENCE: "evidence quote did not check out",
    FLAG_DISAGREEMENT: "the two graders disagreed",
    FLAG_SPREAD: "one grader disagreed with itself across samples",
    FLAG_SINGLE_MODEL: "single-model run, nothing cross-checked",
    FLAG_EXTREME: "0 or 100 -- worth a look either way",
}

VETO_TEXT = {
    "illegible": "the model could not read this",
    "not_found": "this answer is not in the file",
    "unreadable_pdf": "the PDF itself could not be opened",
}


@dataclass
class ReviewItem:
    """One gradeable thing belonging to one student: the unit of review."""

    student_id: str
    row: int                       # index into data["grades"], the write target
    item_index: int                # position within the row's grade list
    key: str                       # "a" / "3" -- how the item is addressed
    label: str                     # "part (a)" / "Q3" -- how it is shown

    grade: Optional[int] = None
    model_grades: Dict[str, Optional[int]] = field(default_factory=dict)
    reasoning: Dict[str, str] = field(default_factory=dict)

    legibility: Optional[str] = None
    presence: Optional[str] = None
    evidence: str = ""
    evidence_ok: Optional[bool] = None
    veto: Optional[str] = None
    confidence: Optional[str] = None
    spread: Optional[int] = None
    samples: List[Optional[int]] = field(default_factory=list)

    tag: Optional[str] = None
    weight: float = 1.0            # marks this item is worth
    single_model: bool = False

    decided: bool = False          # a human has ruled on it
    decision: Optional[Dict[str, Any]] = None

    # ---- derived signals -----------------------------------------------------

    @property
    def disagreement(self) -> Optional[int]:
        """Gap between the two graders, or None if there is no second opinion."""
        present = [g for g in self.model_grades.values() if g is not None]
        if len(present) < 2:
            return None
        return max(present) - min(present)

    def flags(self, conflict_limit: int = 10) -> List[str]:
        """Every reason this item might want a human, most serious first."""
        out: List[str] = []
        if self.veto:
            out.append(FLAG_VETO)
        if self.grade is None and not self.veto:
            out.append(FLAG_NO_GRADE)
        if self.evidence_ok is False:
            out.append(FLAG_EVIDENCE)
        gap = self.disagreement
        if gap is not None and gap > conflict_limit:
            out.append(FLAG_DISAGREEMENT)
        if self.spread is not None and self.spread > conflict_limit:
            out.append(FLAG_SPREAD)
        if self.single_model and not out:
            out.append(FLAG_SINGLE_MODEL)
        if self.grade in (0, 100) and FLAG_VETO not in out:
            out.append(FLAG_EXTREME)
        return out

    def reasoning_for(self, model: str) -> Tuple[str, bool]:
        """
        (text, is_sliced) -- this part's reasoning if it can be isolated.

        The flag matters: a reviewer needs to know whether they are reading the
        model's argument about part (c) or the first third of an argument about
        everything, because only one of those justifies the mark in front of
        them.
        """
        full = self.reasoning.get(model, "") or ""
        section = slice_reasoning(full, self.key)
        return (section, True) if section else (full, False)

    def pages(self) -> List[int]:
        """Pages the model cited for this part, best-effort, nearest first."""
        for model in self.reasoning:
            section, sliced = self.reasoning_for(model)
            found = cited_pages(section if sliced else "")
            if found:
                return found
        for text in self.reasoning.values():
            found = cited_pages(text)
            if found:
                return found
        return []

    def uncertainty(self, conflict_limit: int = 10) -> float:
        """
        How much of a claim this grade is not entitled to make, 0..1.

        Deliberately not called "confidence": it is an ordering heuristic, not a
        probability, and nothing downstream should treat it as calibrated. What
        it has to get right is only the top of the list.

        The top of the scale is reserved for items that are not grades at all --
        a withheld mark, or a missing one. Those cannot be published whatever a
        human decides, so they outrank every merely-doubtful grade.
        """
        if self.veto or self.grade is None:
            return 1.0
        if self.evidence_ok is False:
            return 0.9

        # Disagreement, whether between two models or between one model and
        # itself, is measured against the same yardstick: a 20-point
        # self-disagreement and a 20-point
        # cross-model disagreement mean the same thing about the grade.
        scale = max(1, conflict_limit * 2)
        signals = [0.0]
        if self.disagreement is not None:
            signals.append(min(1.0, self.disagreement / scale))
        if self.spread is not None:
            signals.append(min(1.0, self.spread / scale))
        score = max(signals)

        if self.legibility == "partial":
            score = max(score, 0.35)
        if self.presence == "unattempted":
            # A blank answer is a real 0, but it is also the shape of a page
            # that failed to scan, so it is worth a glance before publishing.
            score = max(score, 0.30)
        if self.grade in (0, 100):
            # The runbook's own advice for a single-model round is "spot-check
            # the extremes: everyone who got 100 and everyone who got 0 on the
            # hardest part". A 0 is also the shape every silent failure in this
            # project takes -- unreadable handwriting, a missing page, a model
            # that cannot read Persian -- so it earns attention by default
            # rather than by the reviewer remembering to go looking.
            score = max(score, 0.40)
        if self.single_model and self.spread is None:
            # Nothing cross-checked this at all. Not evidence of error, but not
            # evidence of correctness either, and the artifact says so.
            score = max(score, 0.25)
        return max(score, 0.02)

    def swing(self) -> float:
        """
        How far this grade could plausibly move, in percentage points.

        Observed disagreement is the best available estimate -- if two graders
        landed 30 apart, a correction of about 30 is on the table. An item with
        no grade at all could move by the whole part.
        """
        if self.veto or self.grade is None:
            return 100.0
        candidates = [c for c in (self.disagreement, self.spread) if c is not None]
        if candidates:
            return float(max(candidates))
        return 100.0 if self.evidence_ok is False else 10.0

    def expected_correction(self, conflict_limit: int = 10) -> float:
        """Marks this item is expected to move if a human looks at it."""
        return self.uncertainty(conflict_limit) * (self.swing() / 100.0) * self.weight

    def priority(self, conflict_limit: int = 10) -> float:
        """
        Queue position: `uncertainty x marks_at_stake`.

        A contested grade on a 20-mark part matters ten times more than a
        contested one on a 2-mark part, and neither has anything to do with the
        student's ID -- which is what the previous reviewer sorted by.
        """
        return self.expected_correction(conflict_limit)


# --- Reading the model's prose ------------------------------------------------

# The grader is asked to reason part by part and does, with a heading per part:
# "**Part (a)**", "Part (b):", and the Persian labels the prompt maps them to.
# Both the heading words and the native labels come from the locale packs.


def slice_reasoning(text: str, key: str) -> str:
    """
    The paragraph of the model's reasoning that concerns one part.

    The grader writes one block of prose covering every part, and the old
    reviewer showed all of it in a disabled 400px text area. Reviewing part (c)
    then means scrolling past two parts you have already decided, on every
    student, which is how a reviewer stops reading and starts clicking.

    Returns "" when no per-part structure is found, and the caller shows the
    whole thing -- guessing a slice would be worse than not slicing.
    """
    if not text:
        return ""
    from . import locales

    locs = locales.active()
    native_labels = locs.part_label_map()
    matches = list(locales.part_heading_regex(locs).finditer(text))
    if not matches:
        return ""
    wanted = str(key).strip().lower()
    for i, match in enumerate(matches):
        label = match.group("label").strip().lower()
        label = native_labels.get(label, label)
        if label != wanted:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        return text[match.start():end].strip()
    return ""


_PAGE_MENTION = re.compile(
    r"pages?\s+((?:\d+\s*(?:,|and|و|-)?\s*)+)", re.IGNORECASE
)


def cited_pages(text: str) -> List[int]:
    """
    Page numbers the model says it found the answer on.

    Worth parsing because it is the difference between a reviewer paging through
    a 17-page scan and landing on the right page immediately. It is also a check
    in its own right: a model citing page 4 for an answer that is on page 9 was
    not reading the pages.
    """
    pages: List[int] = []
    for match in _PAGE_MENTION.finditer(text or ""):
        for number in re.findall(r"\d+", match.group(1)):
            value = int(number)
            if 0 < value < 200 and value not in pages:
                pages.append(value)
    return pages


# --- Loading ------------------------------------------------------------------

def _as_assurance(row: Dict[str, Any], key: str) -> Dict[str, Any]:
    """
    The assurance block for one item, tolerating files written before it existed.

    Old results files resume and print rather than crash everywhere else in this
    codebase (`_result_tags` does the same for `part_constructs`); a review tool
    that could not open the 81-row HW3 file would be reviewing nothing.
    """
    block = row.get("assurance") or {}
    if isinstance(block, dict):
        entry = block.get(key)
        if isinstance(entry, dict):
            return entry
    return {}


def _as_tag(row: Dict[str, Any], key: str) -> Optional[str]:
    for field_name in ("part_tags", "part_constructs"):
        block = row.get(field_name) or {}
        if isinstance(block, dict) and key in block:
            return block[key]
    return None


def _model_block(row: Dict[str, Any], name: str) -> Dict[str, Any]:
    block = row.get(name)
    return block if isinstance(block, dict) else {}


def _grade_at(block: Dict[str, Any], index: int) -> Optional[int]:
    grades = block.get("grades")
    if isinstance(grades, list) and index < len(grades):
        return grades[index]
    if "grade" in block and index == 0:
        return block.get("grade")
    return None


def course_weights(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Marks per item from course.toml's `[[questions]] points`, so a fresh
    review starts from the course's real weighting instead of 1 mark each.
    `points` is a number, split evenly over a question's parts, or a table
    of marks per part (`points = {a = 5, b = 10}`). Weights set in the
    console are kept in the review state and win over these.
    """
    from . import config

    def total(q: int) -> Optional[float]:
        pts = config.QUESTIONS.get(q, {}).get("points")
        if isinstance(pts, dict):
            return float(sum(float(v) for v in pts.values()))
        return None if pts is None else float(pts)

    fmt = detect_format(data)
    out: Dict[str, float] = {}
    if fmt == FORMAT_PARTS:
        q = int(data.get("question") or 0)
        parts = [str(p) for p in (data.get("parts") or [])]
        pts = config.QUESTIONS.get(q, {}).get("points")
        if isinstance(pts, dict):
            out = {str(k): float(v) for k, v in pts.items() if str(k) in parts}
        elif pts is not None and parts:
            out = {p: float(pts) / len(parts) for p in parts}
    elif fmt == FORMAT_MULTI:
        count = int(data.get("num_questions") or 0)
        for q in range(1, count + 1):
            if total(q) is not None:
                out[str(q)] = total(q)
    else:
        q = int(data.get("question") or 1)
        if total(q) is not None:
            out[str(q)] = total(q)
    return out


def build_items(
    data: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> List[ReviewItem]:
    """Flatten a results file into one ReviewItem per gradeable thing."""
    fmt = detect_format(data)
    rows = data.get("grades", [])
    weights = weights or {}
    items: List[ReviewItem] = []

    for row_index, row in enumerate(rows):
        student_id = row.get("student_id", f"row{row_index}")
        single = bool(row.get("single_model") or data.get("single_model"))
        m1, m2 = _model_block(row, "model_1"), _model_block(row, "model_2")
        # A single-model run records model_2 as a name of None. Presenting that
        # as a grader that returned nothing would read as a failed second
        # opinion rather than an opinion that was never sought (ADR-005).
        model_names = [m1.get("name") or "model 1"]
        if not single and m2.get("name"):
            model_names.append(m2["name"])

        if fmt == FORMAT_PARTS:
            keys = list(row.get("parts") or data.get("parts") or [])
            labels = [f"part ({k})" for k in keys]
            finals = row.get("final_grades") or []
        elif fmt == FORMAT_MULTI:
            count = data.get("num_questions") or len(row.get("final_grades") or [])
            keys = [str(i + 1) for i in range(count)]
            labels = [f"Q{i + 1}" for i in range(count)]
            finals = row.get("final_grades") or []
        else:
            q = data.get("question", 1)
            keys, labels = [str(q)], [f"Q{q}"]
            finals = [row.get("final_grade")]

        for i, key in enumerate(keys):
            assurance = _as_assurance(row, key)
            grades = {}
            for slot, block in (("model_1", m1), ("model_2", m2)):
                name = block.get("name")
                if not name:
                    continue
                if slot == "model_2" and single:
                    continue
                grades[name] = _grade_at(block, i)

            items.append(ReviewItem(
                student_id=student_id,
                row=row_index,
                item_index=i,
                key=key,
                label=labels[i],
                grade=finals[i] if i < len(finals) else None,
                model_grades=grades,
                reasoning={
                    b.get("name") or slot: b.get("reasoning", "")
                    for slot, b in (("model 1", m1), ("model 2", m2))
                    if b.get("name") and not (slot == "model 2" and single)
                },
                legibility=assurance.get("legibility"),
                presence=assurance.get("presence"),
                evidence=assurance.get("evidence", "") or "",
                evidence_ok=assurance.get("evidence_ok"),
                veto=assurance.get("veto"),
                confidence=assurance.get("confidence"),
                spread=assurance.get("spread"),
                samples=assurance.get("samples") or [],
                tag=_as_tag(row, key),
                weight=float(weights.get(key, 1.0)),
                single_model=single,
                decided=bool((row.get("reviewed") or {}).get(key)),
                decision=(row.get("reviewed") or {}).get(key),
            ))

        # Nothing above depends on model_names beyond the single-model check;
        # kept explicit so the intent survives a later edit.
        del model_names

    return items


# --- Resolving ----------------------------------------------------------------

def pick_higher(*grades: Optional[int]) -> Optional[int]:
    """
    The higher of the grades on offer, or None if none were offered.

    Returning None rather than 0 in the empty case is the entire point of this
    function existing. The version it replaces ended `... if present else 0`,
    so "accept the higher grade" on a part neither model could grade wrote a
    zero against a student whose work nobody had looked at -- the null-vs-zero
    substitution ADR-003 forbids, reached through the review tool instead of
    through the grader.
    """
    present = [g for g in grades if g is not None]
    return max(present) if present else None


def suggestions(item: ReviewItem) -> List[Tuple[str, Optional[int]]]:
    """Candidate resolutions to offer as one-click buttons, best first."""
    out: List[Tuple[str, Optional[int]]] = []
    present = [g for g in item.model_grades.values() if g is not None]
    if item.samples:
        values = sorted(v for v in item.samples if v is not None)
        if values:
            median = values[len(values) // 2]
            out.append((f"median of {len(values)} samples", median))
    if len(present) >= 2:
        lo, hi = min(present), max(present)
        out.append((f"higher ({hi})", hi))
        out.append((f"lower ({lo})", lo))
        mid = round((lo + hi) / 2)
        if mid not in (lo, hi):
            out.append((f"midpoint ({mid})", mid))
    elif len(present) == 1:
        out.append((f"the one grade given ({present[0]})", present[0]))
    if item.veto or item.grade is None:
        out.append(("withhold -- no grade", None))
    return out


# --- The session --------------------------------------------------------------

REVIEWED_KEY = "reviewed"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write via a temp file and os.replace, per ADR-010."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


class ReviewSession:
    """
    A grades file, its review decisions, and the sidecar state around both.

    Three files are involved and they have different lifetimes:

    - `grades_*.json`     the grader's output, edited in place as marks are set
    - `grades_*.review.jsonl`  an append-only ledger of every human decision
    - `grades_*.review.json`   sidecar state: item weights, sign-off, reviewer

    The ledger is separate and append-only because it is the only artifact here
    with value beyond the round: it is a growing set of human-corrected grades,
    which is the ground truth needed to measure accuracy at all.
    """

    def __init__(self, path: str, reviewer: str = ""):
        self.path = Path(path)
        self.reviewer = reviewer
        self.data: Dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        from .config import RESULTS_SCHEMA_VERSION

        version = int(self.data.get("schema_version", 1))
        if version > RESULTS_SCHEMA_VERSION:
            raise ValueError(
                f"{path} was written by a newer ta-grader (results schema {version}); "
                f"this version reads up to {RESULTS_SCHEMA_VERSION}. Upgrade to review it."
            )
        self.mtime = self.path.stat().st_mtime
        self.format = detect_format(self.data)
        self.state_path = self.path.with_suffix(".review.json")
        self.ledger_path = self.path.with_suffix(".review.jsonl")
        self.state = self._load_state()
        self.items = build_items(self.data, self.weights())

    def weights(self) -> Dict[str, float]:
        """Marks per item: course.toml points, overridden by the console."""
        return {**course_weights(self.data), **(self.state.get("weights") or {})}

    # ---- state ---------------------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"weights": {}, "signed_off": None, "conflict_limit": None}

    def save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    @property
    def conflict_limit(self) -> int:
        return int(
            self.state.get("conflict_limit")
            or self.data.get("conflict_limit")
            or 10
        )

    @property
    def item_keys(self) -> List[str]:
        seen: List[str] = []
        for item in self.items:
            if item.key not in seen:
                seen.append(item.key)
        return seen

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Marks each part is worth. Changes the queue order, nothing else."""
        self.state["weights"] = {k: float(v) for k, v in weights.items()}
        self.save_state()
        for item in self.items:
            item.weight = float(weights.get(item.key, 1.0))

    def file_changed_on_disk(self) -> bool:
        """
        True if something else wrote the grades file since it was loaded.

        The realistic case is a grader still running in another terminal. The
        old reviewer held the whole file in session state and rewrote it on
        every save, so a resumed run's new students were silently deleted by
        the next override.
        """
        try:
            return self.path.stat().st_mtime > self.mtime
        except OSError:
            return False

    def reload(self) -> None:
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.mtime = self.path.stat().st_mtime
        self.items = build_items(self.data, self.weights())

    # ---- queue ---------------------------------------------------------------

    def queue(self, include_decided: bool = False) -> List[ReviewItem]:
        """Every item that wants attention, most valuable first."""
        limit = self.conflict_limit
        pending = [
            i for i in self.items
            if include_decided or not i.decided
        ]
        return sorted(pending, key=lambda i: -i.priority(limit))

    def triage(self, top_n: Optional[int] = None) -> Dict[str, Any]:
        """
        What reviewing the top of the queue is expected to be worth.

        The number is a heuristic sum of per-item estimates and is presented as
        one. Its job is to answer "is it worth carrying on?" -- which no
        currently-shipped view answers at all.
        """
        limit = self.conflict_limit
        q = self.queue()
        head = q[:top_n] if top_n else q
        return {
            "pending": len(q),
            "reviewing": len(head),
            "expected_marks": sum(i.expected_correction(limit) for i in head),
            "students": len({i.student_id for i in head}),
            "blocking": sum(1 for i in head if i.grade is None or i.veto),
        }

    def flag_counts(self) -> Dict[str, int]:
        limit = self.conflict_limit
        counts: Dict[str, int] = {}
        for item in self.items:
            for flag in item.flags(limit):
                counts[flag] = counts.get(flag, 0) + 1
        return counts

    # ---- decisions -----------------------------------------------------------

    def apply(
        self,
        item: ReviewItem,
        new_grade: Optional[int],
        reason: str = "",
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a human's ruling on one item: file, ledger, and in-memory model.

        `new_grade=None` is a legitimate ruling -- it means "this is not a grade
        and must not be published as one", which is what an illegible page or a
        missing part deserves. It is not the same as leaving the item undecided.
        """
        if new_grade is not None:
            new_grade = max(0, min(100, int(new_grade)))
        row = self.data["grades"][item.row]
        before = item.grade

        if self.format == FORMAT_SINGLE:
            row["final_grade"] = new_grade
            row["conflict"] = False
        else:
            finals = row.setdefault("final_grades", [])
            while len(finals) <= item.item_index:
                finals.append(None)
            finals[item.item_index] = new_grade
            conflicts = row.setdefault("conflicts", [False] * len(finals))
            while len(conflicts) <= item.item_index:
                conflicts.append(False)
            conflicts[item.item_index] = False
            row["has_conflict"] = any(conflicts)
            if self.format == FORMAT_PARTS:
                grades_by_key = row.setdefault("part_grades", {})
                grades_by_key[item.key] = new_grade
            row["total_grade"] = sum(g for g in finals if g is not None)

        # An overridden grade keeps its veto in the record. The veto describes
        # what the model saw, and a human deciding the mark anyway does not make
        # the page legible retroactively -- it makes the mark the human's.
        row["manual_override"] = True
        reviewed = row.setdefault(REVIEWED_KEY, {})
        entry = {
            "grade": new_grade,
            "was": before,
            "reason": reason,
            "note": note,
            "by": self.reviewer or os.environ.get("USER", "unknown"),
            "at": _now(),
        }
        reviewed[item.key] = entry

        item.grade = new_grade
        item.decided = True
        item.decision = entry

        self._append_ledger({
            "file": self.path.name,
            "student_id": item.student_id,
            "item": item.key,
            "label": item.label,
            "was": before,
            "now": new_grade,
            "models": item.model_grades,
            "spread": item.spread,
            "veto": item.veto,
            "confidence": item.confidence,
            "evidence": item.evidence[:400],
            "reason": reason,
            "note": note,
            "by": entry["by"],
            "at": entry["at"],
            # A blind mark carries the model's grade beside the human's, taken
            # before the human saw it. That pair is the calibration data; the
            # same pair taken after the reveal is only an agreement rate.
            **(extra or {}),
        })
        self.save()

    def accept(self, item: ReviewItem, reason: str = "accepted as graded") -> None:
        """Confirm the model's mark unchanged. Still a decision, still recorded."""
        self.apply(item, item.grade, reason=reason)

    def undo(self, item: ReviewItem) -> None:
        """Reopen an item. The ledger keeps the original decision regardless."""
        row = self.data["grades"][item.row]
        reviewed = row.get(REVIEWED_KEY) or {}
        entry = reviewed.pop(item.key, None)
        if entry is not None:
            restored = entry.get("was")
            if self.format == FORMAT_SINGLE:
                row["final_grade"] = restored
            else:
                row["final_grades"][item.item_index] = restored
                if self.format == FORMAT_PARTS:
                    row.setdefault("part_grades", {})[item.key] = restored
                row["total_grade"] = sum(
                    g for g in row["final_grades"] if g is not None
                )
            item.grade = restored
            self._append_ledger({
                "file": self.path.name,
                "student_id": item.student_id,
                "item": item.key,
                "was": entry.get("grade"),
                "now": restored,
                "reason": "undo",
                "by": self.reviewer or os.environ.get("USER", "unknown"),
                "at": _now(),
            })
        item.decided = False
        item.decision = None
        self.save()

    def _append_ledger(self, record: Dict[str, Any]) -> None:
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def ledger(self) -> List[Dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        out = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def save(self) -> None:
        counts = [r for r in self.data.get("grades", [])]
        self.data["total_students"] = len(counts)
        if self.format == FORMAT_SINGLE:
            self.data["conflicts"] = sum(1 for r in counts if r.get("conflict"))
        else:
            self.data["students_with_conflicts"] = sum(
                1 for r in counts if r.get("has_conflict")
            )
        atomic_write_json(self.path, self.data)
        self.mtime = self.path.stat().st_mtime

    # ---- sign-off ------------------------------------------------------------

    def blocking_items(self) -> List[ReviewItem]:
        """
        Items that cannot be published whatever else happens.

        A withheld or missing grade is not a low mark, it is an absent one. It
        has to become a number, or an explicit decision not to grade, before a
        batch means anything.
        """
        return [
            i for i in self.items
            if (i.grade is None or i.veto) and not i.decided
        ]

    def sign_off(self, note: str = "") -> Dict[str, Any]:
        """
        A human accepting the batch: assistant, not authority.

        Export is gated on this, so "the TA is the author of record" is a step
        someone takes rather than a sentence in a README.
        """
        record = {
            "at": _now(),
            "by": self.reviewer or os.environ.get("USER", "unknown"),
            "note": note,
            "reviewed": sum(1 for i in self.items if i.decided),
            "items": len(self.items),
            "unresolved": len(self.blocking_items()),
        }
        self.state["signed_off"] = record
        self.save_state()
        self._append_ledger({"event": "sign_off", **record})
        return record

    @property
    def signed_off(self) -> Optional[Dict[str, Any]]:
        return self.state.get("signed_off")

    # ---- fairness ------------------------------------------------------------

    def inconsistencies(
        self, similarity: float = 0.9, gap: int = 15
    ) -> List[Tuple[ReviewItem, ReviewItem, float]]:
        """
        Pairs of near-identical answers that received different marks.

        Those are grading errors by definition and need no ground truth to
        detect, which makes this the one quality check that is persuasive to
        someone who does not trust the model at all: two students wrote the same
        query and got different marks, or they did not.

        This is the cheap version of a consistency audit. The full version compares
        transcriptions, which do not exist yet; this compares the verbatim
        EVIDENCE quotes, which do -- on any run made after `assurance.py`
        landed. On older files it finds nothing and says so rather than
        pretending the cohort was consistent.
        """
        out = []
        by_key: Dict[str, List[ReviewItem]] = {}
        for item in self.items:
            if item.evidence and len(item.evidence) >= 20 and item.grade is not None:
                by_key.setdefault(item.key, []).append(item)
        for group in by_key.values():
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    if abs((a.grade or 0) - (b.grade or 0)) < gap:
                        continue
                    ratio = difflib.SequenceMatcher(
                        None, _normalise(a.evidence), _normalise(b.evidence)
                    ).ratio()
                    if ratio >= similarity:
                        out.append((a, b, ratio))
        return sorted(out, key=lambda t: -abs((t[0].grade or 0) - (t[1].grade or 0)))

    # ---- export --------------------------------------------------------------

    def export_csv(self, path: str) -> str:
        """
        The export that used to be a `python -c` heredoc in the runbook.

        `utf-8-sig` is not decoration: without the BOM Excel mangles the Persian
        filenames used as IDs for submissions that arrived without one.

        A withheld grade exports as an empty cell, never as 0. A spreadsheet is
        exactly where the null-vs-zero distinction gets destroyed, and the
        column that says why is the defence.
        """
        keys = self.item_keys
        by_student: Dict[str, Dict[str, ReviewItem]] = {}
        for item in self.items:
            by_student.setdefault(item.student_id, {})[item.key] = item

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["student_id", *keys, "total_marks", "reviewed", "unresolved", "notes"]
            )
            for student_id in sorted(by_student):
                items = by_student[student_id]
                cells, total, unresolved, notes = [], 0.0, [], []
                for key in keys:
                    item = items.get(key)
                    if item is None or item.grade is None:
                        cells.append("")
                        unresolved.append(key)
                        if item is not None and item.veto:
                            notes.append(f"{key}: {VETO_TEXT.get(item.veto, item.veto)}")
                        continue
                    cells.append(item.grade)
                    # Grades are percentages; weights are marks. The total is
                    # marks, which is the number that reaches the student.
                    total += item.grade / 100.0 * item.weight
                reviewed = sum(1 for i in items.values() if i.decided)
                writer.writerow([
                    student_id, *cells, round(total, 2),
                    f"{reviewed}/{len(items)}",
                    ",".join(unresolved),
                    "; ".join(notes),
                ])
        return path


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# --- Finding the submissions --------------------------------------------------

def find_pdf(student_id: str, folders: Optional[Sequence[str]] = None) -> Optional[Path]:
    """
    Locate a student's PDF across per-assignment folders.

    The old reviewer hardcoded `students_pdf/`, so every one of the 81 HW3
    submissions -- graded out of `students_pdf_hw3/` -- rendered as "PDF not
    found". Showing the page next to the grade is the entire purpose of a review
    tool; it was broken for the only round this project has ever run.
    """
    from .config import student_id_for

    for folder in folders or candidate_pdf_folders():
        candidate = Path(folder) / f"{student_id}.pdf"
        if candidate.exists():
            return candidate
        for pdf in Path(folder).glob("*.pdf"):
            if student_id_for(pdf) == student_id:
                return pdf
    return None


def candidate_pdf_folders(root: Optional[str] = None) -> List[str]:
    """
    Every folder a submission might be in: the configured submissions folder
    and its per-assignment subfolders, plus the older `students_pdf*` folders
    beside the results.
    """
    from . import config

    found = []
    sub = Path(config.PDF_FOLDER)
    if sub.is_dir():
        found.append(sub)
        found += [p for p in sub.iterdir() if p.is_dir()]
    base = Path(root) if root else config.ROOT
    if base.is_dir():
        found += [p for p in base.iterdir()
                  if p.is_dir() and p.name.startswith("students_pdf")]
    return sorted({str(p) for p in found})
