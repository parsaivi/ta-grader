"""
Central configuration.

Every module reads its API settings, model names and folder paths from here so
there is one place to change them. Values come from three layers, highest first:

1. environment variables (usually via a `.env` file),
2. `course.toml` in the course workspace (see `ta-grader init`),
3. the defaults below.

The workspace is the nearest directory at or above the current one that holds
a `course.toml`. Without one, everything resolves against the current directory,
which is how the project ran before it had workspaces.

The API key is deliberately *not* readable from `course.toml`: that file is
meant to be shared between the TAs of a course, and a key in it would be too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

PACKAGE_DATA = Path(__file__).resolve().parent / "data"
COURSE_FILE_NAME = "course.toml"


def find_workspace(start: Optional[Path] = None) -> Optional[Path]:
    """The nearest directory at or above `start` containing a course.toml."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / COURSE_FILE_NAME).is_file():
            return folder
    return None


WORKSPACE: Optional[Path] = None
ROOT: Path = Path.cwd()
COURSE: Dict[str, Any] = {}


def _env_or(key: str, section: str, name: str, default: Any) -> Any:
    """Environment first, then course.toml [section].name, then the default."""
    if os.getenv(key) not in (None, ""):
        return os.getenv(key)
    value = COURSE.get(section, {}).get(name)
    return default if value is None else value


def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def _folder(key: str, name: str, default: str) -> str:
    value = str(_env_or(key, "paths", name, default))
    if WORKSPACE is not None and not os.path.isabs(value):
        return str(WORKSPACE / value)
    return value


def load(start: Optional[Path] = None) -> None:
    """
    (Re)read configuration. Runs once at import; the CLI calls it again after
    resolving the workspace, and tests call it to point at a temporary one.

    Modules that did `from .config import X` keep the value from import time,
    so the CLI resolves the workspace (and chdirs into it) before importing
    anything else.
    """
    global WORKSPACE, ROOT, COURSE
    global API_KEY, API_URL, MODEL_1_NAME, MODEL_2_NAME, VISION_GRADER_1, VISION_GRADER_2
    global VISION_MODEL, SPLITTER_MODEL, MAX_WORKERS, GRADE_CONFLICT_LIMIT, GRADE_RESOLUTION
    global TAG_SCHEME, QUESTION_PARTS, POSTURE, GRADE_SAMPLES, MIN_CHARS_PER_PAGE
    global MIN_ANSWERED_FRACTION, PDF_FOLDER, EXTRACTED_FOLDER, QUESTIONS_FOLDER
    global ANSWERS_FOLDER, TAGSETS_FOLDER, POSTURES_FOLDER, LOCALES_FOLDER, RESULTS_FOLDER
    global OUTPUT_FILE, LANGUAGES, STUDENT_ID_PATTERN, QUESTIONS, MAX_IMAGE_MB

    WORKSPACE = find_workspace(start)
    ROOT = WORKSPACE or (start or Path.cwd()).resolve()
    COURSE = {}
    if WORKSPACE is not None:
        load_dotenv(WORKSPACE / ".env")
        with open(WORKSPACE / COURSE_FILE_NAME, "rb") as f:
            COURSE = tomllib.load(f)
    load_dotenv()

    # --- API -----------------------------------------------------------------
    API_KEY = os.getenv("LLM_KEY")
    API_URL = _env_or("LLM_URL", "api", "url", "https://openrouter.ai/api/v1")

    # --- Models --------------------------------------------------------------
    # Model ids are provider-prefixed for OpenRouter ("openai/...", "google/...").
    # Point LLM_URL at any other OpenAI-compatible endpoint (OpenAI, vLLM,
    # Ollama, a university gateway) and name its models instead.
    #
    # The two graders used for dual-model verification should be *different*
    # models: agreement between two independent models is the whole point.
    # Both must accept image input, since the vision route sends PDF pages.
    # These defaults were chosen by measurement on a Persian/English course,
    # not by list price -- re-measure with `ta-grader benchmark` for yours.
    MODEL_1_NAME = _env_or("MODEL_1", "models", "grader_1", "openai/gpt-4.1-nano")
    MODEL_2_NAME = _env_or("MODEL_2", "models", "grader_2", "google/gemini-2.5-flash")

    # The vision route handles the handwritten minority -- the hardest
    # submissions and only a few percent of a class. Point these at stronger
    # models if handwriting grading looks unreliable; the cost barely moves.
    VISION_GRADER_1 = _env_or("VISION_GRADER_1", "models", "vision_grader_1", MODEL_1_NAME)
    VISION_GRADER_2 = _env_or("VISION_GRADER_2", "models", "vision_grader_2", MODEL_2_NAME)

    # Describes images/diagrams during text extraction.
    VISION_MODEL = _env_or("VISION_MODEL", "models", "figures", "google/gemini-2.5-flash")
    # Locates where each question's answer starts. Text-only work.
    SPLITTER_MODEL = _env_or("SPLITTER_MODEL", "models", "splitter", "google/gemini-2.5-flash")

    # --- Grading behaviour ---------------------------------------------------
    MAX_WORKERS = int(_env_or("MAX_WORKERS", "grading", "workers", 5))

    # Two models' grades further apart than this (on the 0-100 scale) are
    # flagged for human review instead of being silently resolved.
    GRADE_CONFLICT_LIMIT = int(_env_or("GRADE_CONFLICT_LIMIT", "grading", "conflict_limit", 10))

    # How to pick the final grade when the two models agree closely:
    # "max" (generous), "mean" (rounded average) or "min" (strict).
    GRADE_RESOLUTION = str(_env_or("GRADE_RESOLUTION", "grading", "resolution", "max"))

    # Optional course-specific labels recorded alongside each grade. A name
    # from tagsets/ or a path. Empty means no tagging. Tags never affect grades.
    TAG_SCHEME = str(_env_or("TAG_SCHEME", "grading", "tags", ""))

    # Part labels to grade separately within one question, e.g. "a,b,c".
    QUESTION_PARTS = ",".join(_as_list(_env_or("QUESTION_PARTS", "grading", "parts", "")))

    # How to make the marginal calls the marking scheme does not settle. A
    # name from postures/ or a path. Empty uses the built-in defaults, which
    # are still stated in full in the prompt and recorded in the results.
    POSTURE = str(_env_or("POSTURE", "grading", "posture", ""))

    # Grade each submission N times with model 1; the spread is the confidence.
    GRADE_SAMPLES = int(_env_or("GRADE_SAMPLES", "grading", "samples", 1))

    # Pages are sent as JPEG and the DPI stepped down until a submission fits.
    MAX_IMAGE_MB = float(_env_or("MAX_IMAGE_MB", "grading", "max_image_mb", 18))

    # --- Routing between the text and vision graders --------------------------
    # Below this many extracted characters per page, a PDF is scanned or
    # handwritten and the text route would grade it as blank.
    MIN_CHARS_PER_PAGE = int(_env_or("MIN_CHARS_PER_PAGE", "routing", "min_chars_per_page", 400))
    # If the splitter finds answers for fewer than this fraction of questions,
    # the text extracted but is not usable: route to vision.
    MIN_ANSWERED_FRACTION = float(
        _env_or("MIN_ANSWERED_FRACTION", "routing", "min_answered_fraction", 0.25)
    )

    # --- Languages and students ------------------------------------------------
    # Locale packs (data/locales/<code>.toml) to enable. They supply question
    # heading keywords, digit sets, right-to-left repair and the prompt's
    # description of the submission's language.
    LANGUAGES = _as_list(_env_or("LANGUAGES", "course", "languages", "en"))

    # Regex applied to a submission's filename stem; group "id" (or the whole
    # match) becomes the student_id. Empty keeps the whole stem.
    STUDENT_ID_PATTERN = str(_env_or("STUDENT_ID_PATTERN", "students", "id_pattern", ""))

    # Per-question settings from course.toml [[questions]]: id, points, parts,
    # rubric. Optional; questions not listed use the folder conventions.
    QUESTIONS = {int(q["id"]): q for q in COURSE.get("questions", []) if "id" in q}

    # --- Folders -------------------------------------------------------------
    PDF_FOLDER = _folder("PDF_FOLDER", "submissions", "students_pdf")
    EXTRACTED_FOLDER = _folder("EXTRACTED_FOLDER", "extracted", "students_extracted")
    QUESTIONS_FOLDER = _folder("QUESTIONS_FOLDER", "questions", "questions")
    ANSWERS_FOLDER = _folder("ANSWERS_FOLDER", "rubrics", "answers")
    TAGSETS_FOLDER = _folder("TAGSETS_FOLDER", "tagsets", "tagsets")
    POSTURES_FOLDER = _folder("POSTURES_FOLDER", "postures", "postures")
    LOCALES_FOLDER = _folder("LOCALES_FOLDER", "locales", "locales")
    RESULTS_FOLDER = _folder("RESULTS_FOLDER", "results", ".")

    OUTPUT_FILE = os.path.join(RESULTS_FOLDER, os.getenv("OUTPUT_FILE", "grades_result.json"))


# Written by the splitter when a student did not answer a question. Kept as a
# constant so the grader can recognise it instead of sending the literal
# placeholder to the model as if it were an answer.
NO_ANSWER_MARKER = "[NO ANSWER PROVIDED]"


def data_dirs(kind: str) -> List[Path]:
    """
    Where to look for named data files of one kind ("postures", "tagsets",
    "locales"): the workspace's own folder first, so a course can override a
    shipped file by name, then the files shipped with the package.
    """
    own = Path({"postures": POSTURES_FOLDER, "tagsets": TAGSETS_FOLDER,
                "locales": LOCALES_FOLDER}[kind])
    if not own.is_absolute():
        own = ROOT / own
    return [own, PACKAGE_DATA / kind]


def find_data_file(kind: str, name: str, ext: str = ".json") -> Optional[Path]:
    for folder in data_dirs(kind):
        path = folder / f"{name}{ext}"
        if path.is_file():
            return path
    return None


def list_data_files(kind: str, ext: str = ".json") -> List[str]:
    names = set()
    for folder in data_dirs(kind):
        if folder.is_dir():
            names.update(p.stem for p in folder.glob(f"*{ext}"))
    return sorted(names)


def rubric_file(question_num: int, answers_folder: Optional[str] = None) -> Path:
    """
    The marking scheme for one question: the `rubric` named for it in
    course.toml, else `q<N>.md`, else the original `correct_answer_<N>.md`.
    When none exists, returns the path errors should name.
    """
    folder = Path(answers_folder or ANSWERS_FOLDER)
    named = QUESTIONS.get(question_num, {}).get("rubric")
    candidates = []
    if named:
        candidates.append(Path(named) if os.path.isabs(named) else ROOT / named)
    candidates += [folder / f"q{question_num}.md", folder / f"correct_answer_{question_num}.md"]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0] if named else candidates[-1]


def question_points(question_num: int) -> Optional[float]:
    """Points a question is worth, if course.toml says. Grades stay 0-100."""
    value = QUESTIONS.get(question_num, {}).get("points")
    return None if value is None else float(value)


def question_parts(question_num: int) -> List[str]:
    return _as_list(QUESTIONS.get(question_num, {}).get("parts"))


def student_id_for(path) -> str:
    """
    The student id a submission file belongs to. The filename stem by default;
    with `[students] id_pattern`, the regex's `id` group (or whole match), so
    an LMS export named `HW3_402100001_attempt2.pdf` can still key on the id.
    A stem the pattern does not match keeps its full stem rather than failing,
    so one oddly named file never silently merges into another student.
    """
    import re

    stem = Path(path).stem.strip()
    if not STUDENT_ID_PATTERN:
        return stem
    match = re.search(STUDENT_ID_PATTERN, stem)
    if not match:
        return stem
    return match.groupdict().get("id") or match.group(0)


def result_path(name: str) -> str:
    """Where a results file goes: the workspace's results folder."""
    return name if os.path.isabs(name) else os.path.join(RESULTS_FOLDER, name)


def get_client():
    """Return an OpenAI-compatible client pointed at the configured endpoint."""
    from openai import OpenAI

    if not API_KEY:
        raise RuntimeError(
            "LLM_KEY is not set. Put it in the workspace's .env file:\n"
            "    LLM_KEY=your_api_key_here"
        )
    return OpenAI(api_key=API_KEY, base_url=API_URL)


load()
