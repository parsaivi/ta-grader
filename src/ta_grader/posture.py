"""
Grading posture: the marginal-call rules that decide *how* to mark.

A posture is not the marking scheme and not a tag scheme.  The marking scheme
says what the right answer is; a tag scheme records an observation that must not
change the grade.  A posture decides the recurring judgement calls that sit in
between -- which band to pick when an answer falls between two, whether a nearly
right answer gets full marks, whether a syntax slip costs anything.

Two design decisions here are deliberate and worth not undoing.

**Postures select from a fixed vocabulary; they cannot write prompt text.**
Every parameter has a closed set of values, and the sentence sent to the model
for each value lives in this file.  A posture file that could contribute free
text would be a channel for smuggling in course-specific grading rules -- and
those belong in `answers/correct_answer_N.md`, where a human reviews them before
the run, not in a settings file nobody reads.  Same split, same reason as
ADR-006.

**Every parameter is always stated, including the defaults.**  It is tempting to
emit only the non-default choices and keep the prompt short.  Don't: the failure
this project has already had is that a model left without an instruction invents
one.  Told to notice CTEs, it started deducting for them unasked.  Silence about
presentation or language is not neutrality -- it is an invitation to freelance.
So `presentation: ignore` and `language: ignore` produce emphatic prose rather
than nothing at all.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import POSTURES_FOLDER, find_data_file, list_data_files


@dataclass(frozen=True)
class Parameter:
    """One posture knob: a closed set of values, each with its prompt text."""

    key: str
    title: str
    default: str
    values: Dict[str, str]  # value -> the sentence the model is given

    @property
    def names(self) -> List[str]:
        return list(self.values)


# The vocabulary. Order matters only for readability of the prompt.
PARAMETERS: List[Parameter] = [
    Parameter(
        key="boundary_rule",
        title="Answers between two marking bands",
        default="nearest",
        values={
            "up": "When an answer sits between two marking bands, award the "
                  "HIGHER band.",
            "nearest": "When an answer sits between two marking bands, award "
                       "whichever band it is closer to. Do not systematically "
                       "favour the higher or the lower one.",
            "down": "When an answer sits between two marking bands, award the "
                    "LOWER band.",
        },
    ),
    Parameter(
        key="near_miss",
        title="Answers that are materially correct but not identical",
        default="band_down",
        values={
            "full": "An answer that is materially correct -- it would produce "
                    "the right result -- receives FULL marks, even if it "
                    "differs from the reference solution in wording, "
                    "formatting, variable naming, or the particular valid "
                    "construct it uses. Only a difference that changes the "
                    "result may cost marks.",
            "band_down": "An answer that is materially correct but differs from "
                         "the reference solution in some inessential way may "
                         "lose a small number of marks, but must stay within "
                         "one band of full marks.",
            "proportional": "Grade an answer against the reference solution on "
                            "its merits; a materially correct answer that "
                            "differs from the reference is marked by how much "
                            "of the required work it demonstrates.",
        },
    ),
    Parameter(
        key="approach_vs_result",
        title="Right method with the wrong answer, and the reverse",
        default="balanced",
        values={
            "approach": "Credit the method above the result. A correct approach "
                        "carried out with an arithmetic or transcription slip "
                        "keeps most of its marks; a correct result reached by "
                        "an invalid method earns few.",
            "result": "Credit the result above the method. An answer that gets "
                      "the right result scores well even if the working is "
                      "unorthodox; a correct approach that produces the wrong "
                      "result loses substantial marks.",
            "balanced": "Weigh method and result together, as the marking bands "
                        "describe. Neither one alone carries the grade.",
        },
    ),
    Parameter(
        key="minor_errors",
        title="Syntax slips, typos and off-by-one errors",
        default="small",
        values={
            "ignore": "Do NOT deduct for syntax errors, typos, missing "
                      "semicolons, misspelled keywords or similar surface "
                      "slips, provided the intent is unambiguous. Grade the "
                      "logic only.",
            "small": "Surface slips -- syntax errors, typos, a missing "
                     "semicolon -- may cost a few marks at most. They must not "
                     "move an answer more than one band.",
            "full": "Apply the marking scheme to syntax and correctness alike; "
                    "a non-working answer is graded as such.",
        },
    ),
    Parameter(
        key="multiple_attempts",
        title="More than one attempt at the same part",
        default="last",
        values={
            "last": "If the student made several attempts at a part, grade the "
                    "LAST one. Work that is crossed out, struck through or "
                    "otherwise cancelled must be ignored entirely -- do not "
                    "grade it and do not let it lower the grade.",
            "best": "If the student made several attempts at a part, grade the "
                    "BEST one. Work that is crossed out, struck through or "
                    "otherwise cancelled must be ignored entirely.",
            "flag": "If the student made several attempts at a part and it is "
                    "not clear which one counts, say so explicitly in your "
                    "reasoning and grade the last one.",
        },
    ),
    Parameter(
        key="ambiguous",
        title="Answers whose meaning cannot be determined",
        default="flag",
        values={
            "flag": "If you genuinely cannot tell what the student meant, do "
                    "NOT guess a grade in either direction. Say so explicitly "
                    "in your reasoning and mark that part's legibility "
                    "accordingly.",
            "favour": "If you genuinely cannot tell what the student meant, "
                      "resolve the ambiguity in the student's favour.",
            "penalise": "If you genuinely cannot tell what the student meant, "
                        "grade only what is unambiguously demonstrated.",
        },
    ),
    Parameter(
        key="presentation",
        title="Handwriting, neatness and layout",
        default="ignore",
        values={
            "ignore": "Handwriting quality, neatness, crossings-out, layout and "
                      "organisation are NOT graded and must NOT lower any mark. "
                      "A correct answer written messily scores exactly the same "
                      "as a correct answer written neatly. Grade the content "
                      "only. Do not mention presentation as a reason for any "
                      "deduction.",
            "weight": "Presentation and clarity of working form part of the "
                      "marking scheme; apply them as it describes.",
        },
    ),
    Parameter(
        key="language",
        title="The language the answer is written in",
        default="ignore",
        values={
            "ignore": "The language of the answer -- Persian, English, or a "
                      "mixture -- is NOT graded and must NOT affect any mark. "
                      "An answer in Persian and the identical answer in English "
                      "receive the same grade. Do not prefer, and do not "
                      "penalise, either language, and do not mention language "
                      "as a reason for any deduction.",
        },
    ),
]

PARAMETERS_BY_KEY: Dict[str, Parameter] = {p.key: p for p in PARAMETERS}

# Not a prompt parameter: this one decides what a blank part becomes in the
# results file, which is a code path rather than an instruction to the model.
# "null" means a student who attempted nothing gets no grade rather than a 0 --
# some courses want the distinction preserved for a manual decision later.
UNATTEMPTED_VALUES = ("zero", "null")
UNATTEMPTED_DEFAULT = "zero"


@dataclass
class GradingPosture:
    """A named set of marginal-call rules, applied to a whole grading round."""

    key: str
    title: str
    choices: Dict[str, str]
    unattempted: str = UNATTEMPTED_DEFAULT
    description: str = ""  # for humans reading the file; never sent to a model

    def value(self, key: str) -> str:
        return self.choices.get(key, PARAMETERS_BY_KEY[key].default)

    def prompt_section(self) -> str:
        """
        The block of the system prompt that states the posture.

        Every parameter appears, defaults included -- see the module docstring.
        """
        lines = [
            "### Grading posture (how to make the marginal calls)",
            "",
            "These rules decide the judgement calls the marking scheme does not "
            "settle. They apply to every student in this round identically.",
            "",
        ]
        for p in PARAMETERS:
            lines.append(f"- {p.values[self.value(p.key)]}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        """The full posture, for recording in the results file."""
        return {
            "key": self.key,
            "title": self.title,
            "unattempted": self.unattempted,
            "choices": {p.key: self.value(p.key) for p in PARAMETERS},
        }

    def summary(self) -> str:
        """One line, for the run banner."""
        return ", ".join(f"{p.key}={self.value(p.key)}" for p in PARAMETERS)


def default_posture() -> GradingPosture:
    """
    The posture used when none is named.

    This is not "no posture" -- there is no such thing. Grading without stated
    marginal rules just means the model picks its own, differently per student.
    So the default is an explicit, conservative, recorded set.
    """
    return GradingPosture(
        key="default",
        title="Default posture",
        choices={p.key: p.default for p in PARAMETERS},
        unattempted=UNATTEMPTED_DEFAULT,
        description="Conservative defaults, applied when --posture is not given.",
    )


def posture_from_dict(data: Dict[str, Any], default_key: str = "posture") -> GradingPosture:
    choices: Dict[str, str] = {}
    raw = data.get("choices") or {}

    unknown = set(raw) - set(PARAMETERS_BY_KEY)
    if unknown:
        raise ValueError(
            f"Unknown posture parameter(s): {', '.join(sorted(unknown))}. "
            f"Known parameters: {', '.join(PARAMETERS_BY_KEY)}."
        )

    for key, param in PARAMETERS_BY_KEY.items():
        chosen = str(raw.get(key, param.default)).strip().lower()
        if chosen not in param.values:
            raise ValueError(
                f"Posture parameter '{key}' does not accept '{chosen}'. "
                f"Valid values: {', '.join(param.names)}."
            )
        choices[key] = chosen

    unattempted = str(data.get("unattempted", UNATTEMPTED_DEFAULT)).strip().lower()
    if unattempted not in UNATTEMPTED_VALUES:
        raise ValueError(
            f"'unattempted' does not accept '{unattempted}'. "
            f"Valid values: {', '.join(UNATTEMPTED_VALUES)}."
        )

    return GradingPosture(
        key=data.get("key", default_key),
        title=data.get("title", default_key),
        choices=choices,
        unattempted=unattempted,
        description=data.get("description", ""),
    )


def _builtin_path(name: str) -> str:
    found = find_data_file("postures", name)
    return str(found) if found else os.path.join(POSTURES_FOLDER, f"{name}.json")


def load_posture(spec: Optional[str]) -> GradingPosture:
    """
    Load a posture by name (from `postures/`) or by path.

    An empty spec returns the default posture rather than None: unlike tagging,
    posture is never absent, only unstated.
    """
    if not spec:
        return default_posture()

    path = spec if os.path.exists(spec) else _builtin_path(spec)
    if not os.path.exists(path):
        available = ", ".join(sorted(list_postures())) or "(none)"
        raise FileNotFoundError(
            f"Posture '{spec}' not found. Looked for '{path}'. "
            f"Available in {POSTURES_FOLDER}/: {available}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    return posture_from_dict(data, default_key=os.path.splitext(os.path.basename(path))[0])


def list_postures() -> List[str]:
    """Names of the postures shipped in `postures/`."""
    return list_data_files("postures")
