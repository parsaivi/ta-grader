"""
`ta-grader` -- one command for the whole round.

    ta-grader init my-course        make a course workspace
    ta-grader demo                  a synthetic course to try it on
    ta-grader route -q 8            free preview: which students go text, which go vision
    ta-grader grade -q 8            grade (hybrid route by default)
    ta-grader review                open the review console
    ta-grader export results/grades_question_8.json   CSV, once the batch is signed off
    ta-grader benchmark -q 8        measure cost and grades per model on your own PDFs
    ta-grader models                list the models the endpoint serves

Every subcommand runs from the course workspace (the nearest folder holding a
`course.toml`), wherever it was typed. Options after the subcommand are passed
to the grader that runs it; `ta-grader grade --route vision --help` lists them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, config

PART_ONLY_FLAGS = ("--parts", "--tags", "--posture", "--samples", "--single")


def _enter_workspace() -> None:
    """Resolve the workspace, run from its root, and reload config there."""
    workspace = config.find_workspace()
    if workspace is not None:
        os.chdir(workspace)
    config.load()


def _delegate(main, prog: str, argv: List[str]) -> int:
    """Run a module's own argparse `main()` with the remaining arguments."""
    saved = sys.argv
    sys.argv = [prog, *argv]
    try:
        main()
    except SystemExit as exit_:  # argparse --help and usage errors
        return int(exit_.code or 0)
    finally:
        sys.argv = saved
    return 0


# --- init ----------------------------------------------------------------------

COURSE_TEMPLATE = """\
# Course workspace for ta-grader. Share this file with your fellow TAs;
# keep the API key in .env, which is never shared.

[course]
name = "{name}"
# Locale packs for the languages submissions are written in. Shipped: en, fa.
# They set question-heading words, digits, right-to-left repair, and how the
# prompt describes the submission's language.
languages = ["en"]

[api]
# Any OpenAI-compatible endpoint: OpenRouter, OpenAI, vLLM, Ollama, ...
url = "https://openrouter.ai/api/v1"

[models]
# Two *different* models. Agreement between two runs of one model means nothing.
# Both must read images. Measured defaults; re-check with `ta-grader benchmark`.
grader_1 = "openai/gpt-4.1-nano"
grader_2 = "google/gemini-2.5-flash"
splitter = "google/gemini-2.5-flash"
figures = "google/gemini-2.5-flash"

[grading]
conflict_limit = 10      # grades further apart than this (0-100) go to review
resolution = "max"       # max | mean | min, when the two models agree
posture = "standard"     # lenient | standard | strict, or a path to your own
# tags = "proof_technique"   # optional labels recorded next to grades, never scored
samples = 1              # >1 grades N times and reports the spread as confidence
workers = 5

[routing]
min_chars_per_page = 400       # below this a PDF is treated as handwritten
min_answered_fraction = 0.25   # splitter found fewer answers than this -> vision

[students]
# Regex on the PDF filename; group "id" becomes the student id.
# Empty keeps the whole filename. Example for "HW3_402100001.pdf":
# id_pattern = "(?P<id>\\\\d{{6,}})"
id_pattern = ""

[paths]
submissions = "submissions"   # one PDF per student, or one subfolder per assignment
rubrics = "rubrics"           # q1.md, q2.md, ... (see rubrics/q1.md)
results = "results"
extracted = ".work/extracted"
questions = ".work/questions"

# Optional, per question. `points` converts the 0-100 grade on export.
# [[questions]]
# id = 1
# points = 20
# parts = ["a", "b"]
# rubric = "rubrics/q1.md"
"""

RUBRIC_EXAMPLE = r"""\question
State the question here. Optional, but it helps the model.

\answer
Write the reference answer and, ideally, how marks are split. For example:

- 40: identifies that the loop runs n times
- 40: concludes O(n log n), with the reason
- 20: notes the best case

The more explicit the allocation, the less the two models disagree.
"""

GITIGNORE = """\
# Everything here except course.toml and rubrics/ holds student data or a key.
.env
submissions/
results/
.work/
"""

ENV_EXAMPLE = """\
LLM_KEY=
"""


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.directory).resolve()
    if (target / config.COURSE_FILE_NAME).exists() and not args.force:
        print(f"{target / config.COURSE_FILE_NAME} already exists (use --force to overwrite).")
        return 1
    target.mkdir(parents=True, exist_ok=True)
    for folder in ("submissions", "rubrics", "results"):
        (target / folder).mkdir(exist_ok=True)
    name = args.name or target.name
    (target / config.COURSE_FILE_NAME).write_text(COURSE_TEMPLATE.format(name=name), encoding="utf-8")
    example = target / "rubrics" / "q1.md"
    if not example.exists():
        example.write_text(RUBRIC_EXAMPLE, encoding="utf-8")
    if not (target / ".gitignore").exists():
        (target / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    if not (target / ".env").exists():
        (target / ".env").write_text(ENV_EXAMPLE, encoding="utf-8")
    print(f"Created course workspace at {target}\n")
    print("Next:")
    print(f"  1. Put your API key in {target / '.env'}")
    print(f"  2. Put one PDF per student in {target / 'submissions'} (filename = student id)")
    print(f"  3. Write a marking scheme per question: {target / 'rubrics' / 'q1.md'}")
    print(f"  4. cd {target} && ta-grader route -q 1    # free preview")
    return 0


# --- grading -------------------------------------------------------------------

def cmd_route(rest: List[str]) -> int:
    _enter_workspace()
    from . import hybrid_grader

    return _delegate(hybrid_grader.main, "ta-grader route", [*rest, "--dry-run"])


def cmd_grade(route: str, rest: List[str]) -> int:
    _enter_workspace()
    if route != "vision":
        used = [f for f in PART_ONLY_FLAGS if any(a == f or a.startswith(f + "=") for a in rest)]
        if used:
            print(f"{', '.join(used)} run on the vision route, which reads every page "
                  f"and can quote the work it grades. Add --route vision.")
            return 2
    if route == "hybrid":
        from . import hybrid_grader
        return _delegate(hybrid_grader.main, "ta-grader grade", rest)
    if route == "vision":
        from . import vision_grader
        return _delegate(vision_grader.main, "ta-grader grade --route vision", rest)
    from . import pipeline
    return _delegate(pipeline.main, "ta-grader grade --route text", rest)


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import build

    target = Path(args.directory)
    if (target / config.COURSE_FILE_NAME).exists() and not args.force:
        print(f"{target} already holds a course (use --force to rebuild it).")
        return 1
    root = build(target)
    print(f"Built a synthetic demo course at {root} (7 invented students).\n")
    print("Try, free and with no API key:")
    print(f"  cd {root}")
    print("  ta-grader route -q 1")
    print("\nThen, with LLM_KEY set in .env (a few cents for the class):")
    print("  ta-grader grade --route vision -q 1")
    print("  ta-grader review")
    return 0


def cmd_benchmark(rest: List[str]) -> int:
    _enter_workspace()
    from . import benchmark_models

    return _delegate(benchmark_models.main, "ta-grader benchmark", rest)


def cmd_models(args: argparse.Namespace) -> int:
    _enter_workspace()
    client = config.get_client()
    ids = sorted(m.id for m in client.models.list().data)
    for model_id in ids:
        if not args.filter or args.filter.lower() in model_id.lower():
            print(model_id)
    return 0


# --- review & export -------------------------------------------------------------

def cmd_review(rest: List[str]) -> int:
    _enter_workspace()
    if _missing("streamlit"):
        print("The review console needs Streamlit: pip install 'ta-grader[review]'")
        return 1
    app = Path(__file__).resolve().parent / "review_entry.py"
    theme = Path(__file__).resolve().parent / "data" / "streamlit" / "config.toml"
    # Pin the console's light theme wherever it is launched from; without it
    # Streamlit follows the browser into dark mode and the grade field turns
    # near-white on the console's paper-white surfaces.
    flags = _theme_flags(theme) if theme.exists() else ["--theme.base", "light"]
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *flags, *rest]
    return subprocess.call(cmd)


def _missing(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is None


def _theme_flags(theme: Path) -> List[str]:
    """Turn the shipped .streamlit/config.toml into --section.key flags."""
    with open(theme, "rb") as f:
        data = config.tomllib.load(f)
    flags: List[str] = []

    def walk(prefix: str, values: dict) -> None:
        for key, value in values.items():
            if isinstance(value, dict):
                walk(f"{prefix}{key}.", value)
            else:
                flags.extend([f"--{prefix}{key}", str(value).lower() if isinstance(value, bool) else str(value)])

    walk("", data)
    return flags


def cmd_export(args: argparse.Namespace) -> int:
    _enter_workspace()
    from .review_store import ReviewSession

    session = ReviewSession(args.results)
    # No mark reaches a student until a human has accepted the batch -- the
    # same gate the review console puts in front of its export button.
    if not session.signed_off:
        print("Not exported: this batch has not been signed off. "
              "Review it with `ta-grader review` and sign off on the Sign-off tab.")
        return 1
    out = args.output or str(Path(args.results).with_suffix(".csv"))
    try:
        path = session.export_csv(out)
    except Exception as err:  # export refuses an unsigned batch; say why
        print(f"Not exported: {err}")
        return 1
    print(f"Wrote {path}")
    return 0


# --- entry point -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ta-grader",
        description="Grade homework PDFs with two LLMs; review only what they disagree on.",
    )
    parser.add_argument("--version", action="version", version=f"ta-grader {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("init", help="create a course workspace")
    p.add_argument("directory", nargs="?", default=".")
    p.add_argument("--name", help="course name (default: the folder name)")
    p.add_argument("--force", action="store_true", help="overwrite an existing course.toml")

    p = sub.add_parser("demo", help="build a synthetic course to try the tool on")
    p.add_argument("directory", nargs="?", default="ta-grader-demo")
    p.add_argument("--force", action="store_true")

    sub.add_parser("route", help="preview text/vision routing; no API calls", add_help=False)

    p = sub.add_parser("grade", help="grade submissions", add_help=False)
    p.add_argument("--route", choices=["hybrid", "vision", "text"], default="hybrid")

    sub.add_parser("review", help="open the review console", add_help=False)

    p = sub.add_parser("export", help="write a signed-off batch to CSV")
    p.add_argument("results", help="a grades_*.json file")
    p.add_argument("--output", "-o", help="CSV path (default: beside the results file)")

    sub.add_parser("benchmark", help="compare models on your own submissions", add_help=False)

    p = sub.add_parser("models", help="list the models the endpoint serves")
    p.add_argument("filter", nargs="?", help="only ids containing this text")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)

    if args.command == "init":
        return cmd_init(args)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "route":
        return cmd_route(rest)
    if args.command == "grade":
        return cmd_grade(args.route, rest)
    if args.command == "review":
        return cmd_review(rest)
    if args.command == "export":
        return cmd_export(args)
    if args.command == "benchmark":
        return cmd_benchmark(rest)
    if args.command == "models":
        return cmd_models(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
