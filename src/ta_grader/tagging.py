"""
Tag schemes: recording a course-specific observation alongside each grade.

A tag is a label the grader attaches to an answer that is *not* part of the
grade -- "this query uses a CTE", "this proof is by induction", "this code is
recursive".  Courses want different vocabularies, so the vocabulary lives in a
JSON file rather than in the prompt builder.  Point `--tags` at one and the
grader learns the labels; leave it off and no tagging happens at all.

The one thing a scheme cannot configure is whether tags affect the grade.  They
never do.  The boilerplate saying so is written by `prompt_section()` and is not
overridable, because a model told to notice a construct will otherwise quietly
start deducting for it -- that is a real failure this project has already had.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import TAGSETS_FOLDER, find_data_file, list_data_files


@dataclass(frozen=True)
class TagLabel:
    name: str
    description: str = ""
    marker: str = "?"  # single char for the per-student summary grid


@dataclass
class TagScheme:
    """A named vocabulary of labels the grader may attach to each item."""

    key: str                       # results key: "part_tags" carries this scheme
    title: str                     # human name, e.g. "SQL constructs"
    field: str                     # output line the model writes, e.g. "CONSTRUCTS"
    labels: List[TagLabel]
    aliases: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    instruction: str = ""          # optional extra line in the task list

    @property
    def names(self) -> List[str]:
        return [lb.name for lb in self.labels]

    @property
    def markers(self) -> Dict[Optional[str], str]:
        m: Dict[Optional[str], str] = {lb.name: lb.marker for lb in self.labels}
        m[None] = "?"  # not reported -- distinct from any positive label
        return m

    def example(self, count: int) -> str:
        """A plausible example array, so the model sees the exact shape wanted."""
        names = self.names or ["none"]
        return ", ".join((names * count)[:count])

    def prompt_section(self, item_word: str) -> str:
        """The block of the system prompt that defines the vocabulary."""
        lines = [
            f"### {self.title} (recorded alongside the grade, NOT part of the grade)",
            f"For each {item_word}, report exactly one of:",
        ]
        for lb in self.labels:
            lines.append(f"- `{lb.name}`" + (f" -- {lb.description}" if lb.description else ""))
        if self.notes:
            lines.append("")
            lines.extend(self.notes)
        lines += [
            "",
            "This label is descriptive bookkeeping only. It is NOT an error, is",
            "NOT penalised, and must NOT lower the grade. An answer that carries",
            "any of these labels and is otherwise perfect scores 100. Grade exactly",
            "as you would if these labels had never been mentioned.",
        ]
        return "\n".join(lines)

    def output_spec(self, count: int, ordering: str) -> str:
        """The lines describing the tag array in the required output format."""
        quoted = ", ".join(f"`{n}`" for n in self.names)
        return (
            f"{self.field}: [{self.example(count)}]\n"
            f"\nThe {self.field} array MUST contain exactly {count} entries, in "
            f"{ordering}, and use only the labels {quoted}."
        )


def _builtin_path(name: str) -> str:
    found = find_data_file("tagsets", name)
    return str(found) if found else os.path.join(TAGSETS_FOLDER, f"{name}.json")


def load_tag_scheme(spec: Optional[str]) -> Optional[TagScheme]:
    """
    Load a tag scheme by name (from `tagsets/`) or by path.

    Returns None for an empty spec, which means "do not tag anything" -- the
    prompt then never mentions tags and the results carry no tag keys.
    """
    if not spec:
        return None

    path = spec if os.path.exists(spec) else _builtin_path(spec)
    if not os.path.exists(path):
        available = ", ".join(sorted(list_tag_schemes())) or "(none)"
        raise FileNotFoundError(
            f"Tag scheme '{spec}' not found. Looked for '{path}'. "
            f"Available in {TAGSETS_FOLDER}/: {available}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    return tag_scheme_from_dict(data, default_key=os.path.splitext(os.path.basename(path))[0])


def tag_scheme_from_dict(data: Dict[str, Any], default_key: str = "tags") -> TagScheme:
    raw_labels = data.get("labels") or []
    if not raw_labels:
        raise ValueError("A tag scheme must define at least one label.")

    labels = [
        TagLabel(
            name=str(lb["name"]).strip().lower(),
            description=lb.get("description", ""),
            marker=(lb.get("marker") or "?")[:1],
        )
        for lb in raw_labels
    ]

    names = [lb.name for lb in labels]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate label names in tag scheme: {names}")

    return TagScheme(
        key=data.get("key", default_key),
        title=data.get("title", "Tags"),
        field=data.get("field", "TAGS").upper(),
        labels=labels,
        aliases={str(k).strip().lower(): str(v).strip().lower()
                 for k, v in (data.get("aliases") or {}).items()},
        notes=list(data.get("notes") or []),
        instruction=data.get("instruction", ""),
    )


def list_tag_schemes() -> List[str]:
    """Names of the schemes shipped in `tagsets/`."""
    return list_data_files("tagsets")
