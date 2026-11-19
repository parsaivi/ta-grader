"""course.toml workspaces, the CLI's init, and the id/rubric conventions."""

import json
from pathlib import Path

import pytest

from ta_grader import cli, config


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    assert cli.main(["init", str(tmp_path / "course"), "--name", "Algorithms"]) == 0
    root = tmp_path / "course"
    monkeypatch.chdir(root)
    config.load()
    yield root
    monkeypatch.chdir("/")
    config.load()


def test_init_creates_a_workspace_that_keeps_student_data_out_of_git(workspace):
    assert (workspace / "course.toml").is_file()
    assert (workspace / "rubrics" / "q1.md").is_file()
    ignored = (workspace / ".gitignore").read_text()
    for entry in (".env", "submissions/", "results/", ".work/"):
        assert entry in ignored


def test_init_refuses_to_overwrite(workspace):
    assert cli.main(["init", str(workspace)]) == 1


def test_course_toml_is_read_and_paths_resolve_in_the_workspace(workspace):
    assert config.WORKSPACE == workspace.resolve()
    assert config.COURSE["course"]["name"] == "Algorithms"
    assert Path(config.PDF_FOLDER) == workspace.resolve() / "submissions"
    assert Path(config.result_path("grades_question_1.json")).parent == workspace.resolve() / "results"
    assert config.POSTURE == "standard"


def test_workspace_is_found_from_a_subfolder(workspace, monkeypatch):
    monkeypatch.chdir(workspace / "rubrics")
    config.load()
    assert config.WORKSPACE == workspace.resolve()


def test_environment_overrides_course_toml(workspace, monkeypatch):
    monkeypatch.setenv("GRADE_CONFLICT_LIMIT", "3")
    config.load()
    assert config.GRADE_CONFLICT_LIMIT == 3


def test_api_key_is_never_read_from_course_toml(workspace, monkeypatch):
    monkeypatch.delenv("LLM_KEY", raising=False)
    text = (workspace / "course.toml").read_text()
    (workspace / "course.toml").write_text(text.replace("[api]", '[api]\nkey = "sk-leak"'))
    config.load()
    assert config.API_KEY != "sk-leak"


def test_rubric_lookup_prefers_course_toml_then_q_then_legacy(workspace):
    rubrics = workspace / "rubrics"
    (rubrics / "correct_answer_2.md").write_text("legacy")
    assert config.rubric_file(2).name == "correct_answer_2.md"
    (rubrics / "q2.md").write_text("new")
    assert config.rubric_file(2).name == "q2.md"
    (rubrics / "two.md").write_text("named")
    with open(workspace / "course.toml", "a") as f:
        f.write('\n[[questions]]\nid = 2\npoints = 15\nparts = ["a", "b"]\nrubric = "rubrics/two.md"\n')
    config.load()
    assert config.rubric_file(2).name == "two.md"
    assert config.question_points(2) == 15
    assert config.question_parts(2) == ["a", "b"]


def test_student_id_pattern(workspace, monkeypatch):
    assert config.student_id_for("HW3_402100001.pdf") == "HW3_402100001"
    monkeypatch.setenv("STUDENT_ID_PATTERN", r"(?P<id>\d{6,})")
    config.load()
    assert config.student_id_for("submissions/HW3_402100001_late.pdf") == "402100001"
    # A name the pattern does not match keeps its own stem, never merging into another.
    assert config.student_id_for("odd-name.pdf") == "odd-name"


def test_shipped_postures_and_workspace_overrides(workspace):
    from ta_grader.posture import list_postures, load_posture

    assert {"lenient", "standard", "strict"} <= set(list_postures())
    (workspace / "postures").mkdir()
    data = json.loads((config.PACKAGE_DATA / "postures" / "strict.json").read_text())
    data["key"] = "ours"
    (workspace / "postures" / "ours.json").write_text(json.dumps(data))
    assert "ours" in list_postures()
    assert load_posture("ours").key == "ours"


def test_part_only_flags_are_refused_off_the_vision_route(workspace, capsys):
    assert cli.main(["grade", "-q", "1", "--parts", "a,b"]) == 2
    assert "--route vision" in capsys.readouterr().out


def _parts_results(question=2, parts=("a", "b")):
    return {
        "question": question, "parts": list(parts),
        "grades": [{
            "student_id": "S0001", "question": question, "parts": list(parts),
            "part_grades": {p: 100 for p in parts}, "final_grades": [100] * len(parts),
            "conflicts": [False] * len(parts),
            "model_1": {"name": "m1", "grades": [100] * len(parts), "reasoning": ""},
            "model_2": {"name": "m2", "grades": [100] * len(parts), "reasoning": ""},
        }],
    }


def test_course_points_become_review_weights_and_export_marks(workspace):
    from ta_grader.review_store import ReviewSession, course_weights

    with open(workspace / "course.toml", "a") as f:
        f.write('\n[[questions]]\nid = 2\npoints = {a = 5, b = 15}\n')
    config.load()
    data = _parts_results()
    assert course_weights(data) == {"a": 5.0, "b": 15.0}

    path = workspace / "results" / "grades_question_2_parts.json"
    path.write_text(json.dumps(data))
    session = ReviewSession(str(path))
    assert [i.weight for i in session.items] == [5.0, 15.0]
    session.set_weights({"a": 10.0, "b": 15.0})          # the console wins
    assert ReviewSession(str(path)).weights()["a"] == 10.0


def test_points_split_evenly_over_parts(workspace):
    from ta_grader.review_store import course_weights

    with open(workspace / "course.toml", "a") as f:
        f.write('\n[[questions]]\nid = 2\npoints = 20\n')
    config.load()
    assert course_weights(_parts_results()) == {"a": 10.0, "b": 10.0}
