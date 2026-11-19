"""
Tests for the parts that used to fail silently.

Run with:  python -m pytest test_grading.py -v
(or:       python test_grading.py  for a plain run without pytest)

No API key or network access is needed -- the LLM is stubbed out.
"""

import json
from unittest.mock import MagicMock, patch

from ta_grader import config
from ta_grader.assurance import (
    LEGIBILITY_ALIASES,
    LEGIBILITY_VALUES,
    PRESENCE_ALIASES,
    PRESENCE_VALUES,
    VETO_ILLEGIBLE,
    VETO_NOT_FOUND,
    apply_vetoes,
    check_evidence,
    evidence_echoes_reference,
    summarise_samples,
)
from ta_grader.grade_parsing import (
    extract_evidence,
    extract_grades,
    extract_single_grade,
    extract_tags,
    resolve_grade,
    resolve_grades,
)
from ta_grader.posture import PARAMETERS, load_posture, posture_from_dict
from ta_grader.question_splitter import (
    QuestionSplitter,
    _sanitise_starts,
    find_headings_heuristically,
    number_lines,
    slice_by_starts,
)
from ta_grader.smart_pdf_extractor import fix_rtl_text, is_rtl_line
from ta_grader.tagging import load_tag_scheme, tag_scheme_from_dict

# These tests come from a Persian/English course, and exercise both packs.
config.LANGUAGES = ["en", "fa"]

# --- Grade parsing ------------------------------------------------------------


def test_extract_grade_plain():
    assert extract_single_grade("Good work.\nGRADE: 85") == 85


def test_extract_grade_with_brackets():
    # The old prompt showed "GRADE: [number]" but the regex required a bare
    # digit, so bracketed replies fell through to the number-scanning fallback.
    assert extract_single_grade("GRADE: [85]") == 85


def test_extract_grade_with_markdown_and_slash():
    assert extract_single_grade("**GRADE:** 72/100") == 72
    assert extract_single_grade("grade = 40") == 40


def test_extract_grade_takes_last_mention():
    text = "Initially I thought GRADE: 50, but on reflection\nGRADE: 70"
    assert extract_single_grade(text) == 70


def test_extract_grade_never_invents_from_prose():
    # This is the bug that mattered: "6 transistors" must not become a grade.
    text = "The student describes 6 transistors and 4 resistors. Well argued."
    assert extract_single_grade(text) is None


def test_extract_grade_empty():
    assert extract_single_grade("") is None
    assert extract_single_grade(None) is None


def test_extract_grade_clamped():
    assert extract_single_grade("GRADE: 150") == 100


def test_extract_grades_array():
    assert extract_grades("GRADES: [85, 70, 100]", 3) == [85, 70, 100]


def test_extract_grades_pads_short_array():
    assert extract_grades("GRADES: [85, 70]", 4) == [85, 70, None, None]


def test_extract_grades_truncates_long_array():
    assert extract_grades("GRADES: [1, 2, 3, 4, 5]", 3) == [1, 2, 3]


def test_extract_grades_handles_nulls():
    assert extract_grades("GRADES: [85, null, 30]", 3) == [85, None, 30]


def test_extract_grades_missing():
    assert extract_grades("I could not grade this.", 3) == [None, None, None]


# --- Grade resolution ---------------------------------------------------------


def test_agreement_is_not_a_conflict():
    final, conflict = resolve_grade(80, 85, conflict_limit=10)
    assert (final, conflict) == (85, False)


def test_disagreement_is_flagged():
    final, conflict = resolve_grade(30, 90, conflict_limit=10)
    assert conflict is True


def test_single_model_failure_is_flagged_for_review():
    # A grade that only one model produced has not been cross-checked, so it
    # must not be presented as an agreed result.
    final, conflict = resolve_grade(75, None)
    assert (final, conflict) == (75, True)


def test_total_failure_yields_no_grade():
    final, conflict = resolve_grade(None, None)
    assert final is None  # not 0 -- a 0 would look like a real bad score
    assert conflict is True


def test_resolution_modes():
    assert resolve_grade(60, 70, resolution="max")[0] == 70
    assert resolve_grade(60, 70, resolution="min")[0] == 60
    assert resolve_grade(60, 70, resolution="mean")[0] == 65


def test_zero_is_a_real_grade():
    final, conflict = resolve_grade(0, 0)
    assert (final, conflict) == (0, False)


def test_resolve_grades_handles_ragged_lists():
    finals, conflicts = resolve_grades([80, 90], [80])
    assert finals == [80, 90]
    assert conflicts == [False, True]  # Q2 only had one model's opinion


# --- Splitter -----------------------------------------------------------------


def test_number_lines():
    assert number_lines("a\nb") == "1| a\n2| b"


def test_slice_by_starts():
    text = "intro\nQ1 here\nmore q1\nQ2 here\nmore q2"
    starts = {1: 2, 2: 4}
    result = slice_by_starts(text, starts, 2)
    assert result["1"] == "Q1 here\nmore q1"
    assert result["2"] == "Q2 here\nmore q2"


def test_slice_skips_unanswered_without_truncating_previous():
    # Q2 unanswered: Q1 must run all the way to Q3, not stop early.
    text = "Q1 line\na\nb\nQ3 line\nc"
    starts = {1: 1, 2: None, 3: 4}
    result = slice_by_starts(text, starts, 3)
    assert result["1"] == "Q1 line\na\nb"
    assert result["2"] == ""
    assert result["3"] == "Q3 line\nc"


def test_sanitise_rejects_out_of_order_starts():
    # A hallucinated decreasing line number would produce an empty slice.
    clean = _sanitise_starts({1: 10, 2: 4, 3: 20}, 3, total_lines=50)
    assert clean == {1: 10, 2: None, 3: 20}


def test_sanitise_rejects_out_of_range_starts():
    clean = _sanitise_starts({1: 999, 2: 0, 3: 5}, 3, total_lines=50)
    assert clean == {1: None, 2: None, 3: 5}


def test_heuristic_finds_english_headings():
    text = "cover page\nQuestion 1\nanswer one\nQuestion 2\nanswer two"
    assert find_headings_heuristically(text, 2) == {1: 2, 2: 4}


def test_heuristic_finds_reversed_persian_headings():
    # RTL extraction puts the number before the keyword: "۱ سوال".
    text = "به نام خدا\n۱ سوال\nپاسخ\n۲ سوال\nپاسخ"
    assert find_headings_heuristically(text, 2) == {1: 2, 2: 4}


def test_heuristic_ignores_numbered_lists_inside_image_descriptions():
    # Vision-written figure prose is full of "1." list items that are not
    # question headings.
    text = (
        "[IMAGE DESCRIPTION]: A diagram showing:\n"
        "1. **Top block** connected to a bus\n"
        "2. **Bottom block** with a decoder\n"
        "\n"
        "1\n"
        "the student's real answer\n"
    )
    assert find_headings_heuristically(text, 2) == {1: 5}


def test_heuristic_requires_bare_numbers_to_stand_alone():
    text = "3. this is a list item in a sentence that continues\n"
    assert find_headings_heuristically(text, 5) == {}


def _stub_splitter(reply: str) -> QuestionSplitter:
    """Build a QuestionSplitter whose LLM returns a fixed reply."""
    with patch("ta_grader.question_splitter.get_client") as get_client:
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=reply))]
        )
        get_client.return_value = client
        return QuestionSplitter()


def test_splitter_uses_llm_start_lines():
    text = "cover\nfirst answer\nstill q1\nsecond answer"
    splitter = _stub_splitter(json.dumps({"starts": {"1": 2, "2": 4}}))
    result = splitter.split_homework(text, 2)
    assert result["1"] == "first answer\nstill q1"
    assert result["2"] == "second answer"


def test_splitter_falls_back_to_regex_when_llm_fails():
    # The original code returned empty strings for every question here, which
    # is how a student's entire submission got recorded as unanswered.
    text = "cover\nQuestion 1\nanswer one\nQuestion 2\nanswer two"
    with patch("ta_grader.question_splitter.get_client") as get_client:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        get_client.return_value = client
        splitter = QuestionSplitter()

    with patch("ta_grader.question_splitter.time.sleep"):  # don't wait out the backoff
        result = splitter.split_homework(text, 2)

    assert result["1"] == "Question 1\nanswer one"
    assert result["2"] == "Question 2\nanswer two"


def test_splitter_survives_long_input():
    """
    The original failure mode: a long homework made the model echo more text
    than max_tokens allowed, the JSON truncated, and every question came back
    empty. Start lines are a fixed-size reply, so length is irrelevant.
    """
    text = "\n".join(["Question 1"] + ["filler"] * 5000 + ["Question 2", "answer"])
    splitter = _stub_splitter(json.dumps({"starts": {"1": 1, "2": 5002}}))
    result = splitter.split_homework(text, 2)
    assert len(result["1"]) > 10000
    assert result["2"] == "Question 2\nanswer"


# --- RTL repair ---------------------------------------------------------------


def test_rtl_word_order_restored():
    assert fix_rtl_text("خدا نام به") == "به نام خدا"


def test_latin_text_untouched():
    assert fix_rtl_text("hello world 123") == "hello world 123"


def test_is_rtl_line():
    assert is_rtl_line("سوال اول")
    assert not is_rtl_line("Question one")
    assert not is_rtl_line("12345")


# --- Text/vision routing ------------------------------------------------------


def _assessment(student_id, route, pages=10, chars=10000):
    from pathlib import Path

    from ta_grader.hybrid_grader import Assessment

    return Assessment(student_id, Path(f"{student_id}.pdf"), pages, chars, route, "")


def test_reassess_reroutes_student_the_splitter_could_not_parse(tmp=None):
    """
    Text extracted, but the splitter found almost nothing. That student would
    be graded as blank on the text route, so they must go to vision instead.
    """
    import tempfile
    from pathlib import Path

    from ta_grader.config import NO_ANSWER_MARKER
    from ta_grader.hybrid_grader import TEXT_ROUTE, VISION_ROUTE, reassess_after_split

    with tempfile.TemporaryDirectory() as d:
        for q in range(1, 5):
            qdir = Path(d) / f"question_{q}"
            qdir.mkdir()
            # "good" answered everything; "bad" answered nothing.
            (qdir / "good.txt").write_text("a real answer", encoding="utf-8")
            (qdir / "bad.txt").write_text(NO_ANSWER_MARKER, encoding="utf-8")

        out = reassess_after_split(
            [_assessment("good", TEXT_ROUTE), _assessment("bad", TEXT_ROUTE)],
            d, num_questions=4,
        )
        routes = {a.student_id: a.route for a in out}

    assert routes["good"] == TEXT_ROUTE
    assert routes["bad"] == VISION_ROUTE


def test_reassess_leaves_students_alone_before_split_has_run():
    """With no per-question files on disk there is nothing to judge yet."""
    import tempfile

    from ta_grader.hybrid_grader import TEXT_ROUTE, reassess_after_split

    with tempfile.TemporaryDirectory() as d:
        out = reassess_after_split([_assessment("s1", TEXT_ROUTE)], d, num_questions=4)

    assert out[0].route == TEXT_ROUTE


def test_reassess_never_downgrades_a_vision_student():
    import tempfile

    from ta_grader.hybrid_grader import VISION_ROUTE, reassess_after_split

    with tempfile.TemporaryDirectory() as d:
        out = reassess_after_split([_assessment("s1", VISION_ROUTE)], d, num_questions=4)

    assert out[0].route == VISION_ROUTE


def test_render_stays_within_the_api_image_budget():
    """
    Scanned submissions are the largest and the ones that must use vision. A
    17-page scan at 150 DPI rendered to 31 MB of PNG and the API rejected the
    request outright, failing exactly the students the vision route exists for.
    """
    import base64
    import tempfile
    from pathlib import Path

    import pymupdf as fitz

    from ta_grader.vision_grader import MAX_IMAGE_BYTES, VisionGrader

    # A synthetic PDF of large, noisy pages -- the worst case for compression.
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "big.pdf"
        doc = fitz.open()
        for _ in range(20):
            page = doc.new_page(width=1200, height=1600)
            for i in range(0, 1600, 7):
                page.draw_line(fitz.Point(0, i), fitz.Point(1200, i + 40))
        doc.save(str(path))
        doc.close()

        grader = VisionGrader.__new__(VisionGrader)  # skip API client setup
        grader.dpi = 150

        images = grader.pdf_to_images(str(path))

    raw = sum(len(base64.b64decode(i)) for i in images)
    assert len(images) == 20
    assert raw <= MAX_IMAGE_BYTES, f"{raw} bytes exceeds budget {MAX_IMAGE_BYTES}"


# --- Tag schemes --------------------------------------------------------------


SCHEME = {
    "key": "constructs",
    "title": "SQL construct labels",
    "field": "CONSTRUCTS",
    "labels": [{"name": "none"}, {"name": "cte"}, {"name": "subquery"},
               {"name": "cte+subquery"}],
    "aliases": {"both": "cte+subquery"},
}


def _tags(text, n, scheme=None):
    s = tag_scheme_from_dict(scheme or SCHEME)
    return extract_tags(text, n, s.field, s.names, s.aliases)


def test_extract_tags_basic():
    assert _tags("CONSTRUCTS: [none, cte, subquery]", 3) == \
        ["none", "cte", "subquery"]


def test_extract_tags_applies_aliases():
    assert _tags("CONSTRUCTS: [both, cte]", 2) == ["cte+subquery", "cte"]


def test_extract_tags_discards_unknown_labels():
    # A label outside the scheme is dropped rather than stored, so downstream
    # filters can trust the vocabulary is exhaustive.
    assert _tags("CONSTRUCTS: [window_function, cte]", 2) == [None, "cte"]


def test_extract_tags_missing_line_is_unreported():
    # None means "the model did not say", which is not the same as the label
    # "none", a positive statement that it looked and found nothing.
    assert _tags("GRADES: [90, 80]", 2) == [None, None]


def test_extract_tags_pads_short_array():
    assert _tags("CONSTRUCTS: [cte]", 3) == ["cte", None, None]


def test_extract_tags_uses_scheme_field_not_a_hardcoded_one():
    # The vocabulary and the output line are configurable; a scheme must not
    # pick up another scheme's array.
    other = {"key": "technique", "field": "TECHNIQUES",
             "labels": [{"name": "induction"}, {"name": "direct"}]}
    text = "CONSTRUCTS: [cte, cte]\nTECHNIQUES: [induction, direct]"
    assert _tags(text, 2, other) == ["induction", "direct"]
    assert _tags(text, 2) == ["cte", "cte"]


def test_no_tag_scheme_means_no_tagging():
    assert load_tag_scheme("") is None
    assert load_tag_scheme(None) is None


def test_builtin_scheme_loads():
    s = load_tag_scheme("sql_constructs")
    assert s.field == "CONSTRUCTS"
    assert set(s.names) == {"none", "cte", "subquery", "cte+subquery"}


def test_scheme_prompt_forbids_grade_impact():
    # The "this does not affect the grade" boilerplate is not overridable by a
    # scheme file: a model told to notice a construct starts deducting for it.
    section = load_tag_scheme("sql_constructs").prompt_section("part")
    assert "must NOT lower the grade" in section


def test_parts_prompt_omits_tags_without_a_scheme():
    from ta_grader.vision_grader import get_grading_prompt_parts

    prompt = get_grading_prompt_parts(2, ["a", "b"], "scheme text")
    assert "CONSTRUCTS" not in prompt
    assert "GRADES: [85, 70]" in prompt


def test_parts_prompt_includes_tags_with_a_scheme():
    from ta_grader.vision_grader import get_grading_prompt_parts

    prompt = get_grading_prompt_parts(
        2, ["a", "b"], "scheme text", load_tag_scheme("sql_constructs")
    )
    assert "CONSTRUCTS: [none, cte]" in prompt
    assert "`cte+subquery`" in prompt


# --- Assurance: legibility, presence, evidence, confidence --------------------


def test_illegible_answer_gets_no_grade_not_a_zero():
    # The failure this exists to stop: a model that cannot read Persian
    # handwriting returns 0, which is indistinguishable from a student who
    # wrote nothing and is wrong in the direction that harms them.
    grades, flags, vetoes = apply_vetoes(
        [70, 0, 55], ["clear", "illegible", "clear"], ["answered"] * 3
    )
    assert grades == [70, None, 55]
    assert vetoes == [None, VETO_ILLEGIBLE, None]
    assert flags == [False, True, False]


def test_answer_missing_from_the_file_gets_no_grade_not_a_zero():
    # A page that failed to scan is not a wrong answer.
    grades, flags, vetoes = apply_vetoes(
        [80, 0], ["clear", "clear"], ["answered", "not_found"]
    )
    assert grades == [80, None]
    assert vetoes == [None, VETO_NOT_FOUND]
    assert flags == [False, True]


def test_unattempted_is_a_real_zero_by_default():
    # Distinct from the two above: the answer's place was found and it was
    # blank, so 0 is a claim about the student's work that we can actually make.
    grades, flags, vetoes = apply_vetoes([0], ["clear"], ["unattempted"])
    assert grades == [0]
    assert vetoes == [None]
    assert flags == [False]


def test_unattempted_can_be_withheld_by_posture():
    grades, flags, _ = apply_vetoes(
        [0], ["clear"], ["unattempted"], unattempted="null"
    )
    assert grades == [None]
    assert flags == [True]


def test_not_found_beats_illegible():
    # If it is not there, "could not read it" is not the useful description.
    _, _, vetoes = apply_vetoes([0], ["illegible"], ["not_found"])
    assert vetoes == [VETO_NOT_FOUND]


def test_vetoes_never_raise_a_grade_or_invent_one():
    grades, _, _ = apply_vetoes([None, 40], ["clear", "clear"], ["answered"] * 2)
    assert grades == [None, 40]


def test_unreported_assurance_fields_leave_grades_alone():
    # A model that ignored the new fields entirely must not have its grades
    # thrown away -- that would be a worse regression than the bug being fixed.
    grades, flags, _ = apply_vetoes([70, 0], [None, None], [None, None])
    assert grades == [70, 0]
    assert flags == [False, False]


def test_legibility_and_presence_parse_with_aliases():
    text = "LEGIBILITY: [clear, unreadable, partial]\nPRESENCE: [present, missing, blank]"
    assert extract_tags(text, 3, "LEGIBILITY", LEGIBILITY_VALUES, LEGIBILITY_ALIASES) == [
        "clear", "illegible", "partial",
    ]
    assert extract_tags(text, 3, "PRESENCE", PRESENCE_VALUES, PRESENCE_ALIASES) == [
        "answered", "not_found", "unattempted",
    ]


def test_evidence_quotes_survive_commas():
    # The grade and tag arrays split on commas; a quoted SQL line is full of
    # them, so evidence needs its own parser.
    text = 'EVIDENCE: ["SELECT u, v FROM Edges", "WITH E AS (a, b)", ""]'
    assert extract_evidence(text, 3) == [
        "SELECT u, v FROM Edges", "WITH E AS (a, b)", "",
    ]


def test_evidence_missing_line_is_blank_not_an_error():
    assert extract_evidence("no evidence here", 2) == ["", ""]


def test_evidence_lifted_from_the_marking_scheme_is_caught():
    # gpt-4.1-nano asserted "no subqueries or CTEs" about pages that plainly
    # began "WITH E AS (...)" -- it was reciting the answer key. A quote that
    # turns out to be the answer key is that failure, made mechanical.
    reference = "The expected answer is: WITH E AS (SELECT u, v FROM Edges)"
    ok, problems = check_evidence(
        ["WITH E AS (SELECT u, v FROM Edges)"], [90], reference
    )
    assert ok == [False]
    assert "marking scheme" in problems[0]


def test_genuine_student_quote_passes_evidence_check():
    ok, problems = check_evidence(
        ["SELECT DISTINCT a.u, b.v FROM Edges a JOIN Edges b"],
        [80],
        "reference solution text",
    )
    assert ok == [True]
    assert problems == []


def test_a_grade_with_no_evidence_is_reported():
    ok, problems = check_evidence([""], [75], "reference")
    assert ok == [False]
    assert "no evidence quote" in problems[0]


def test_evidence_is_not_required_where_there_is_no_grade():
    ok, problems = check_evidence([""], [None], "reference")
    assert ok == [None]
    assert problems == []


def test_short_evidence_quotes_are_not_flagged_as_echoes():
    # "SELECT u, v" appears in the reference and in every submission alike.
    assert not evidence_echoes_reference("SELECT", "SELECT u, v FROM Edges")


def test_confidence_uses_median_not_max():
    # GRADE_RESOLUTION=max is right for two agreeing models and wrong across
    # repeated samples of one: the max of N noisy draws is a bias that grows
    # with N, not an estimate.
    grade, spread, level = summarise_samples([20, 40, 20], conflict_limit=10)
    assert grade == 20
    assert spread == 20
    assert level == "medium"


def test_confidence_is_high_only_within_the_conflict_limit():
    assert summarise_samples([80, 85, 80], 10)[2] == "high"
    assert summarise_samples([20, 60, 30], 10)[2] == "low"


def test_a_part_some_samples_failed_to_grade_is_not_high_confidence():
    grade, _, level = summarise_samples([80, None, 82], 10)
    assert grade == 81
    assert level == "medium"


def test_single_sample_reports_no_confidence():
    # The single-sample path must behave exactly as it did before sampling
    # existed: the grade passes through and nothing is claimed about it.
    grade, spread, level = summarise_samples([85], 10)
    assert (grade, spread, level) == (85, None, "unmeasured")


def test_no_samples_at_all_is_no_grade():
    assert summarise_samples([None, None], 10) == (None, None, "unmeasured")


# --- Grading posture ----------------------------------------------------------


def test_posture_is_never_absent():
    # Unlike a tag scheme, an unstated posture is not "no posture" -- it is the
    # model choosing its own marginal rules, differently per student.
    assert load_posture("").key == "default"
    assert load_posture(None).key == "default"


def test_builtin_postures_load():
    assert load_posture("strict").value("boundary_rule") == "down"
    assert load_posture("lenient").value("near_miss") == "full"


def test_posture_always_states_every_parameter():
    # Silence about presentation or language is not neutrality; a model without
    # an instruction invents one. Defaults must be spoken aloud.
    section = load_posture("lenient").prompt_section()
    for param in PARAMETERS:
        assert param.values[load_posture("lenient").value(param.key)] in section


def test_posture_cannot_penalise_presentation_or_language_by_default():
    for name in ("standard", "strict", "lenient"):
        section = load_posture(name).prompt_section()
        assert "Handwriting quality" in section
        assert "must NOT affect any mark" in section


def test_posture_rejects_unknown_parameters():
    # A posture file cannot contribute free text or invent knobs; course rules
    # that change grades belong in the marking scheme where they are reviewed.
    try:
        posture_from_dict({"choices": {"penalise_ctes": "yes"}})
    except ValueError as exc:
        assert "penalise_ctes" in str(exc)
    else:
        raise AssertionError("unknown posture parameter was accepted")


def test_posture_rejects_values_outside_the_vocabulary():
    try:
        posture_from_dict({"choices": {"boundary_rule": "vibes"}})
    except ValueError as exc:
        assert "vibes" in str(exc)
    else:
        raise AssertionError("invalid posture value was accepted")


def test_posture_records_all_choices_for_the_results_file():
    # Storing only the name would silently misdescribe a run's marks after any
    # later edit to strict.json.
    d = load_posture("strict").as_dict()
    assert d["key"] == "strict"
    assert set(d["choices"]) == {p.key for p in PARAMETERS}


def test_parts_prompt_states_the_posture_and_asks_for_assurance():
    from ta_grader.vision_grader import get_grading_prompt_parts

    prompt = get_grading_prompt_parts(2, ["a", "b"], "scheme", None, load_posture("strict"))
    assert "award the LOWER band" in prompt
    assert "LEGIBILITY: [clear, clear]" in prompt
    assert "PRESENCE: [answered, answered]" in prompt
    assert "EVIDENCE:" in prompt


def test_assess_pdf_routes_unopenable_file_to_vision():
    from pathlib import Path

    from ta_grader.hybrid_grader import VISION_ROUTE, assess_pdf

    a = assess_pdf(Path("definitely_not_a_real_file.pdf"))
    assert a.route == VISION_ROUTE


# --- Review console -----------------------------------------------------------
#
# The reviewer can break every grading invariant the grader defends, because it
# writes the same file. These cover the ones it actually broke.

def _parts_file(tmpdir, rows=None):
    import json
    from pathlib import Path

    data = {
        "question": 2,
        "parts": ["a", "b", "c"],
        "grades": rows if rows is not None else [{
            "student_id": "S0001",
            "question": 2,
            "parts": ["a", "b", "c"],
            "single_model": True,
            "part_grades": {"a": 90, "b": None, "c": 20},
            "final_grades": [90, None, 20],
            "conflicts": [False, True, False],
            "has_conflict": True,
            "assurance": {
                "a": {"legibility": "clear", "presence": "answered",
                      "evidence": "SELECT a,b,c FROM E"},
                "b": {"legibility": "illegible", "presence": "answered", "evidence": "",
                      "veto": "illegible"},
                "c": {"legibility": "clear", "presence": "answered",
                      "evidence": "WITH E AS (...)"},
            },
            "model_1": {"name": "google/gemini-2.5-flash", "grades": [90, None, 20],
                        "reasoning": "**Part (a)**\nGood, page 4.\n\n**Part (b)**\nCannot read.\n\n"
                                     "**Part (c)**\nOn pages 9 and 10, misses checks."},
            "model_2": {"name": None, "grades": [None, None, None],
                        "reasoning": "[single-model run: second grader not called]"},
        }],
    }
    path = Path(tmpdir) / "grades_test.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_per_part_files_are_recognised_as_per_part():
    # The old reviewer tested `question`/`num_questions` first and so rendered
    # the only production round this project has run as "Q1 Q2 Q3".
    from ta_grader.review_store import FORMAT_MULTI, FORMAT_PARTS, FORMAT_SINGLE, detect_format

    assert detect_format({"question": 2, "parts": ["a", "b"]}) == FORMAT_PARTS
    assert detect_format({"question": 2}) == FORMAT_SINGLE
    assert detect_format({"num_questions": 10}) == FORMAT_MULTI


def test_pick_higher_of_nothing_is_none_not_zero():
    # The version this replaces returned 0 here, which turned "neither model
    # graded this" into "the student scored nothing" on one click.
    from ta_grader.review_store import pick_higher

    assert pick_higher(None, None) is None
    assert pick_higher(0, None) == 0
    assert pick_higher(40, 20) == 40


def test_a_withheld_grade_outranks_a_merely_doubtful_one():
    from ta_grader.review_store import ReviewItem

    vetoed = ReviewItem("s", 0, 0, "b", "part (b)", grade=None, veto="illegible")
    doubtful = ReviewItem("s", 0, 1, "c", "part (c)", grade=55,
                          model_grades={"m1": 50, "m2": 60})
    assert vetoed.uncertainty() == 1.0
    assert vetoed.priority() > doubtful.priority()


def test_single_model_runs_do_not_show_a_second_grader():
    import tempfile

    from ta_grader.review_store import ReviewSession

    with tempfile.TemporaryDirectory() as tmp:
        session = ReviewSession(_parts_file(tmp))
        item = next(i for i in session.items if i.key == "a")
        assert list(item.model_grades) == ["google/gemini-2.5-flash"]
        assert item.single_model is True
        assert item.disagreement is None  # not "the second model returned nothing"


def test_withholding_a_grade_writes_null_and_an_empty_cell():
    import csv
    import tempfile
    from pathlib import Path

    from ta_grader.review_store import ReviewSession

    with tempfile.TemporaryDirectory() as tmp:
        session = ReviewSession(_parts_file(tmp), reviewer="tester")
        item = next(i for i in session.items if i.key == "b")
        session.apply(item, None, reason="illegible -- graded by hand")

        row = session.data["grades"][0]
        assert row["final_grades"][1] is None
        assert row["part_grades"]["b"] is None

        out = str(Path(tmp) / "out.csv")
        session.export_csv(out)
        with open(out, encoding="utf-8-sig") as f:
            record = next(iter(csv.DictReader(f)))
        assert record["b"] == ""          # never "0"
        assert "b" in record["unresolved"]


def test_every_decision_lands_in_the_ledger_with_the_original():
    import tempfile

    from ta_grader.review_store import ReviewSession

    with tempfile.TemporaryDirectory() as tmp:
        session = ReviewSession(_parts_file(tmp), reviewer="tester")
        item = next(i for i in session.items if i.key == "c")
        session.apply(item, 45, reason="read the page -- model misread it")

        entry = session.ledger()[-1]
        assert (entry["was"], entry["now"]) == (20, 45)
        assert entry["by"] == "tester"
        assert entry["reason"].startswith("read the page")


def test_a_blind_mark_records_both_numbers():
    # The pair (what the human said before seeing the model, what the model
    # said) is the calibration data doc 11 Stage 0 asks for. The same pair
    # collected after the reveal is only an agreement rate.
    import tempfile

    from ta_grader.review_store import ReviewSession

    with tempfile.TemporaryDirectory() as tmp:
        session = ReviewSession(_parts_file(tmp), reviewer="tester")
        item = next(i for i in session.items if i.key == "c")
        session.apply(item, 45, reason="blind", extra={
            "blind": True, "blind_grade": 45, "model_grade": 20,
        })
        entry = session.ledger()[-1]
        assert entry["blind"] is True
        assert (entry["blind_grade"], entry["model_grade"]) == (45, 20)


def test_review_results_are_written_atomically():
    # ADR-010 applies to the reviewer too: it writes the same file a running
    # grader may be appending to.
    import tempfile
    from unittest.mock import patch

    from ta_grader.review_store import ReviewSession

    with tempfile.TemporaryDirectory() as tmp:
        session = ReviewSession(_parts_file(tmp))
        with patch("ta_grader.review_store.os.replace") as replace:
            session.save()
            assert replace.called


def test_a_concurrent_write_is_noticed_before_it_is_clobbered():
    import json
    import os
    import tempfile
    import time

    from ta_grader.review_store import ReviewSession

    with tempfile.TemporaryDirectory() as tmp:
        path = _parts_file(tmp)
        session = ReviewSession(path)
        assert session.file_changed_on_disk() is False
        data = json.loads(open(path, encoding="utf-8").read())
        data["grades"].append({"student_id": "999", "parts": ["a", "b", "c"],
                               "final_grades": [1, 2, 3], "model_1": {}, "model_2": {}})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        later = time.time() + 5
        os.utime(path, (later, later))
        assert session.file_changed_on_disk() is True


def test_reasoning_is_sliced_to_the_part_under_review():
    from ta_grader.review_store import cited_pages, slice_reasoning

    text = ("**Part (a)**\nGood, page 4.\n\n**Part (b)**\nCannot read.\n\n"
            "**Part (c)**\nOn pages 9 and 10, misses checks.")
    assert slice_reasoning(text, "c").startswith("**Part (c)**")
    assert "Part (b)" not in slice_reasoning(text, "c")
    assert slice_reasoning("no headings here", "a") == ""   # caller shows it whole
    assert cited_pages(slice_reasoning(text, "c")) == [9, 10]


def test_persian_part_headings_slice_too():
    from ta_grader.review_store import slice_reasoning

    text = "بخش (الف)\nfirst\n\nبخش (ب)\nsecond"
    assert "second" in slice_reasoning(text, "b")
    assert "first" not in slice_reasoning(text, "b")


def test_unresolved_items_block_a_clean_sign_off():
    import tempfile

    from ta_grader.review_store import ReviewSession

    with tempfile.TemporaryDirectory() as tmp:
        session = ReviewSession(_parts_file(tmp))
        blocking = session.blocking_items()
        assert [i.key for i in blocking] == ["b"]   # the vetoed part
        record = session.sign_off("published anyway")
        assert record["unresolved"] == 1            # recorded, not hidden


def test_old_files_without_assurance_still_load():
    import tempfile

    from ta_grader.review_store import ReviewSession

    with tempfile.TemporaryDirectory() as tmp:
        path = _parts_file(tmp, rows=[{
            "student_id": "1", "parts": ["a", "b", "c"],
            "part_grades": {"a": 80, "b": 60, "c": 40},
            "part_constructs": {"a": "none", "b": "cte", "c": "subquery"},
            "final_grades": [80, 60, 40], "conflicts": [False] * 3,
            "single_model": True,
            "model_1": {"name": "m", "grades": [80, 60, 40], "reasoning": "x"},
            "model_2": {"name": None, "grades": [None] * 3, "reasoning": ""},
        }])
        session = ReviewSession(path)
        assert len(session.items) == 3
        assert session.items[1].tag == "cte"        # legacy key still read
        assert session.items[0].veto is None


if __name__ == "__main__":
    import sys
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
