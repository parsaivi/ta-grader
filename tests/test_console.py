"""The review console renders on real results without raising."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402
from test_end_to_end import FakeClient, demo  # noqa: E402,F401

from ta_grader import cli  # noqa: E402

ENTRY = Path(__file__).resolve().parents[1] / "src" / "ta_grader" / "review_entry.py"


def test_console_renders_demo_results(demo, monkeypatch):  # noqa: F811
    import ta_grader.vision_grader as vg

    monkeypatch.setattr(vg, "get_client", lambda: FakeClient())
    assert cli.main(["grade", "--route", "vision", "-q", "1",
                     "--model-1", "grader-one", "--model-2", "grader-two"]) == 0
    # The sidecar must not be offered as a results file.
    (demo / "results" / "grades_question_1_parts.review.json").write_text("{}")

    at = AppTest.from_file(str(ENTRY), default_timeout=60).run()
    assert not at.exception
    assert not at.error
    assert len(at.button) > 0
