"""Locale packs: language-specific behaviour comes from data, not code."""

import pytest

from ta_grader import config, locales
from ta_grader.question_splitter import find_headings_heuristically, splitter_system_prompt
from ta_grader.review_store import slice_reasoning


@pytest.fixture
def languages():
    saved = config.LANGUAGES

    def use(*codes):
        config.LANGUAGES = list(codes)

    yield use
    config.LANGUAGES = saved


def test_shipped_packs_load():
    for code in ("en", "fa"):
        loc = locales.load_locale(code)
        assert loc.question_words


def test_unknown_pack_names_the_available_ones():
    with pytest.raises(FileNotFoundError, match="en"):
        locales.load_locale("xx")


def test_english_only_course_does_not_read_persian_headings(languages):
    languages("en")
    text = "intro\nسوال ۱\nanswer one\nQuestion 2\nanswer two"
    assert find_headings_heuristically(text, 2) == {2: 4}


def test_persian_pack_adds_words_digits_and_reversed_order(languages):
    languages("en", "fa")
    text = "intro\nسوال ۱\nanswer one\n۲ سوال\nanswer two"
    assert find_headings_heuristically(text, 2) == {1: 2, 2: 4}


def test_language_sentence_names_every_enabled_language(languages):
    languages("en")
    assert locales.active().language_sentence() == "The submission is written in English."
    languages("fa", "en")
    assert "Persian or English" in locales.active().language_sentence()


def test_native_part_labels_map_in_order(languages):
    languages("en", "fa")
    mapping = locales.active().part_label_map()
    assert mapping["الف"] == "a" and mapping["ج"] == "c"
    assert "(الف)" in locales.active().part_label_sentence(["a", "b"])


def test_part_label_sentence_is_empty_without_native_labels(languages):
    languages("en")
    assert locales.active().part_label_sentence(["a", "b"]) == ""


def test_reasoning_slices_on_native_labels(languages):
    languages("en", "fa")
    text = "**بخش الف**\nfirst\n**بخش ب**\nsecond"
    assert slice_reasoning(text, "b").endswith("second")


def test_splitter_prompt_lists_enabled_languages_only(languages):
    languages("en")
    assert "سوال" not in splitter_system_prompt()
    languages("en", "fa")
    assert "سوال ۱" in splitter_system_prompt()


def test_workspace_pack_overrides_by_name(tmp_path, monkeypatch, languages):
    (tmp_path / "course.toml").write_text('[course]\nlanguages = ["de"]\n')
    (tmp_path / "locales").mkdir()
    (tmp_path / "locales" / "de.toml").write_text(
        'name = "German"\nquestion_words = ["aufgabe"]\n'
    )
    monkeypatch.chdir(tmp_path)
    try:
        config.load()
        assert config.LANGUAGES == ["de"]
        text = "Name\nAufgabe 1\nx\nAufgabe 2\ny"
        assert find_headings_heuristically(text, 2) == {1: 2, 2: 4}
    finally:
        monkeypatch.chdir("/")
        config.load()
