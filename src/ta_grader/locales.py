"""
Locale packs: how submissions in a given language mark questions and parts.

Everything language-specific the grader needs -- question-heading words, native
digits, part labels, right-to-left repair, and how the prompt describes the
language -- lives in `data/locales/<code>.toml` (or the workspace's `locales/`
folder), not in the code. Enabling a language is a line in course.toml;
supporting a new one is a new file.

Several packs are usually enabled at once (a Persian course gets English
answers too), so callers use the merged view `active()` returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from . import config


@dataclass(frozen=True)
class Locale:
    code: str
    name: str
    rtl: bool = False
    question_words: Tuple[str, ...] = ()
    digits: str = ""
    part_words: Tuple[str, ...] = ()
    part_labels: Tuple[str, ...] = ()
    heading_examples: Tuple[str, ...] = ()


def load_locale(code: str) -> Locale:
    path = config.find_data_file("locales", code, ".toml")
    if path is None:
        available = ", ".join(config.list_data_files("locales", ".toml")) or "(none)"
        raise FileNotFoundError(
            f"No locale pack '{code}'. Available: {available}. "
            f"Add one as locales/{code}.toml in the workspace."
        )
    with open(path, "rb") as f:
        data = config.tomllib.load(f)
    digits = str(data.get("digits", ""))
    if len(digits) % 10:
        raise ValueError(f"Locale '{code}': digits must list 0-9, once per digit set.")
    return Locale(
        code=code,
        name=str(data.get("name", code)),
        rtl=bool(data.get("rtl", False)),
        question_words=tuple(str(w).lower() for w in data.get("question_words", [])),
        digits=digits,
        part_words=tuple(str(w).lower() for w in data.get("part_words", [])),
        part_labels=tuple(str(w) for w in data.get("part_labels", [])),
        heading_examples=tuple(str(w) for w in data.get("heading_examples", [])),
    )


@dataclass(frozen=True)
class LocaleSet:
    """The merged view of every enabled pack."""

    locales: Tuple[Locale, ...] = field(default_factory=tuple)

    @property
    def names(self) -> List[str]:
        return [loc.name for loc in self.locales]

    @property
    def rtl(self) -> bool:
        return any(loc.rtl for loc in self.locales)

    def _merged(self, attr: str) -> List[str]:
        seen: List[str] = []
        for loc in self.locales:
            for item in getattr(loc, attr):
                if item not in seen:
                    seen.append(item)
        return seen

    @property
    def question_words(self) -> List[str]:
        return self._merged("question_words")

    @property
    def rtl_question_words(self) -> List[str]:
        return [w for loc in self.locales if loc.rtl for w in loc.question_words]

    @property
    def part_words(self) -> List[str]:
        return self._merged("part_words")

    @property
    def heading_examples(self) -> List[str]:
        return self._merged("heading_examples")

    def digit_table(self) -> Dict[int, int]:
        """str.translate table taking every native digit to 0-9."""
        src = "".join(loc.digits for loc in self.locales)
        return str.maketrans(src, "0123456789" * (len(src) // 10)) if src else {}

    def part_label_map(self) -> Dict[str, str]:
        """Native part label -> a, b, c, ... in order."""
        mapping: Dict[str, str] = {}
        for loc in self.locales:
            for i, label in enumerate(loc.part_labels):
                mapping.setdefault(label, "abcdefghijklmnopqrstuvwxyz"[i])
        return mapping

    def language_sentence(self) -> str:
        """How the grading prompt describes the submission's language."""
        names = self.names
        if not names:
            return "The submission may be in any language."
        if len(names) == 1:
            return f"The submission is written in {names[0]}."
        listed = ", ".join(names[:-1]) + f" or {names[-1]}"
        return f"The submission may be in {listed}, or a mix of them."

    def part_label_sentence(self, labels: Sequence[str]) -> str:
        """Tells the grader which native labels mean which part, if any."""
        sentences = []
        for loc in self.locales:
            native = loc.part_labels[: len(labels)]
            if native:
                shown = ", ".join(f"({x})" for x in native)
                sentences.append(
                    f"{loc.name} part labels {shown} correspond to parts "
                    f"{', '.join(labels[: len(native)])} in that order."
                )
        return " ".join(sentences)


def _alternation(words: Sequence[str]) -> str:
    # Longest first, so "question" wins over "q".
    return "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))


def heading_regex(locales: LocaleSet) -> "re.Pattern[str]":
    """
    An explicit question heading at the start of a line: "Question 3",
    "سوال ۳", "#3". Digits are normalised before matching. For right-to-left
    languages the number may come first ("۱ سوال"): PDFs that store RTL runs
    in visual order extract them that way.
    """
    words = _alternation(locales.question_words) or r"question"
    branches = [rf"(?:{words})\s*[.:\-]?\s*(\d{{1,2}})", r"\#\s*(\d{1,2})"]
    if locales.rtl_question_words:
        branches.append(rf"(\d{{1,2}})\s*[.:\-]?\s*(?:{_alternation(locales.rtl_question_words)})")
    return re.compile(
        r"^\s*(?:" + "|".join(branches) + r")(?:\s|[.:\-)]|$)",
        re.IGNORECASE,
    )


def part_heading_regex(locales: LocaleSet) -> "re.Pattern[str]":
    """A heading the grader writes before each part's reasoning: "**Part (b)**"."""
    words = _alternation(["part", *locales.part_words])
    native = _alternation(list(locales.part_label_map()))
    label = r"[a-zA-Z]|[0-9]{1,2}" + (f"|{native}" if native else "")
    return re.compile(
        rf"^[ \t>*_#]*\**\s*(?:{words})\s*[\(\[]?\s*"
        rf"(?P<label>{label})\s*[\)\]]?\s*\**\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )


@lru_cache(maxsize=None)
def _cached(codes: Tuple[str, ...]) -> LocaleSet:
    return LocaleSet(tuple(load_locale(c) for c in codes))


def active(codes: Sequence[str] = ()) -> LocaleSet:
    """The packs named (default: course.toml's languages)."""
    return _cached(tuple(codes or config.LANGUAGES))
