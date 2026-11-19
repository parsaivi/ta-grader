"""
The whole round on the synthetic demo course, offline: route, grade on the
vision route with a stand-in model client, review, sign off, export.
"""

import csv
import json
from types import SimpleNamespace

import pytest

from ta_grader import cli, config

# Model 1 and model 2 agree on part (a) and disagree by 40 on part (b), so the
# run must flag exactly the (b) items for review.
REPLIES = {
    "grader-one": "**Part (a)**\nCorrect bound and sum.\n**Part (b)**\nRight verdict.\n"
                  "GRADES: [90, 80]\nLEGIBILITY: [clear, clear]\nPRESENCE: [answered, answered]\n"
                  'EVIDENCE: ["The sum is n(n-1)/2", "giving [1, 2b, 2a]"]',
    "grader-two": "**Part (a)**\nFine.\n**Part (b)**\nNo counterexample.\n"
                  "GRADES: [85, 40]\nLEGIBILITY: [clear, clear]\nPRESENCE: [answered, answered]\n"
                  'EVIDENCE: ["The sum is n(n-1)/2", "giving [1, 2b, 2a]"]',
}


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, **kwargs):
        self.calls.append(model)
        message = SimpleNamespace(content=REPLIES[model])
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100, cost=0.001)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage
        )


@pytest.fixture
def demo(tmp_path, monkeypatch):
    assert cli.main(["demo", str(tmp_path / "demo")]) == 0
    root = tmp_path / "demo"
    monkeypatch.chdir(root)
    monkeypatch.setenv("LLM_KEY", "test-key")
    monkeypatch.setenv("MODEL_1", "grader-one")
    monkeypatch.setenv("MODEL_2", "grader-two")
    config.load()
    yield root
    monkeypatch.chdir("/")
    config.load()


def test_route_is_free_and_sends_the_image_only_pdf_to_vision(demo, capsys):
    assert cli.main(["route", "-q", "1", "-n", "1"]) == 0
    out = capsys.readouterr().out
    assert "vision route : 1" in out and "S1007" in out


def test_full_round_offline(demo, monkeypatch):
    fake = FakeClient()
    import ta_grader.vision_grader as vg

    monkeypatch.setattr(vg, "get_client", lambda: fake)
    # course.toml names parts a,b for question 1, so no --parts is needed.
    assert cli.main(["grade", "--route", "vision", "-q", "1",
                     "--model-1", "grader-one", "--model-2", "grader-two"]) == 0

    results = demo / "results" / "grades_question_1_parts.json"
    data = json.loads(results.read_text())
    assert len(data["grades"]) == 7
    assert set(fake.calls) == {"grader-one", "grader-two"}   # both graders ran
    row = data["grades"][0]
    assert row["final_grades"] == [90, 80]                    # max of agreeing pair; b conflicted
    assert row["conflicts"] == [False, True]
    assert data["posture"]["key"] == "standard"               # posture recorded in full

    from ta_grader.review_store import ReviewSession

    session = ReviewSession(str(results), reviewer="tester")
    assert {i.key for i in session.queue()} >= {"b"}
    assert [i.weight for i in session.items[:2]] == [6.0, 4.0]   # points from course.toml

    # Export is refused before sign-off.
    assert cli.main(["export", str(results)]) == 1

    for item in session.items:
        if item.key == "b":
            session.apply(item, 60, reason="checked counterexample")
    student = session.items[0].student_id
    withheld = next(i for i in session.items if i.student_id == student and i.key == "a")
    session.apply(withheld, None, reason="withheld for a resubmission")
    session.sign_off("demo batch")

    assert cli.main(["export", str(results)]) == 0
    rows = list(csv.DictReader(open(results.with_suffix(".csv"), encoding="utf-8-sig")))
    first = next(r for r in rows if r["student_id"] == student)
    assert first["a"] == ""                       # withheld is empty, never 0
    other = next(r for r in rows if r["student_id"] != student)
    assert float(other["total_marks"]) == pytest.approx(0.9 * 6 + 0.6 * 4)

    ledger = [json.loads(line) for line in open(results.with_suffix(".review.jsonl"))]
    assert any(e.get("event") == "sign_off" for e in ledger)
