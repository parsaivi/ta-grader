"""
Text-based homework grader.

Grades a question from already-extracted, already-split text answers (see
pipeline.py steps 1 and 2).  Cheaper than vision_grader.py because the PDF
pages are never re-sent to the model, but it depends on extraction quality.

Both models grade independently; disagreement beyond GRADE_CONFLICT_LIMIT is
flagged for human review rather than silently resolved.
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from .config import (
    ANSWERS_FOLDER,
    GRADE_CONFLICT_LIMIT,
    GRADE_RESOLUTION,
    MAX_WORKERS,
    MODEL_1_NAME,
    MODEL_2_NAME,
    NO_ANSWER_MARKER,
    OUTPUT_FILE,
    QUESTIONS_FOLDER,
    RESULTS_SCHEMA_VERSION,
    get_client,
    result_path,
    rubric_file,
)
from .grade_parsing import extract_single_grade, resolve_grade
from .grading_prompts import GRADING_SYSTEM_PROMPT

# Re-exported for pipeline.py and older callers.
__all__ = [
    "HomeworkGrader",
    "load_student_answers",
    "load_question_and_answer",
    "load_existing_results",
    "save_results",
    "GRADE_CONFLICT_LIMIT",
    "MAX_WORKERS",
]


class HomeworkGrader:
    def __init__(
        self,
        model_1: str = MODEL_1_NAME,
        model_2: str = MODEL_2_NAME,
        conflict_limit: int = GRADE_CONFLICT_LIMIT,
        resolution: str = GRADE_RESOLUTION,
    ):
        self.client = get_client()
        self.model_1 = model_1
        self.model_2 = model_2
        self.conflict_limit = conflict_limit
        self.resolution = resolution

    def call_llm(self, model_name: str, system_prompt: str, user_content: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            print(f"Error calling LLM ({model_name}): {exc}")
            return ""

    def grade_with_model(
        self, model_name: str, question: str, correct_answer: str, student_answer: str
    ) -> Dict[str, Any]:
        user_content = f"""### Question:
{question}

### Correct Answer:
{correct_answer}

### Student's Answer:
{student_answer}
"""
        raw_output = self.call_llm(model_name, GRADING_SYSTEM_PROMPT, user_content)
        return {
            "model": model_name,
            "grade": extract_single_grade(raw_output),
            "raw_output": raw_output,
        }

    def grade_student(
        self, student_id: str, question: str, correct_answer: str, student_answer: str
    ) -> Dict[str, Any]:
        """Grade one student's answer with both models."""
        # An unanswered question is a certain zero. Spending two API calls to
        # confirm that -- and risking the model inventing partial credit for a
        # placeholder string -- is pure waste.
        if not student_answer.strip() or student_answer.strip() == NO_ANSWER_MARKER:
            return {
                "student_id": student_id,
                "final_grade": 0,
                "conflict": False,
                "model_1": {"name": self.model_1, "grade": 0,
                            "reasoning": "[No answer submitted -- not sent to the model.]"},
                "model_2": {"name": self.model_2, "grade": 0,
                            "reasoning": "[No answer submitted -- not sent to the model.]"},
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(
                self.grade_with_model, self.model_1, question, correct_answer, student_answer
            )
            f2 = pool.submit(
                self.grade_with_model, self.model_2, question, correct_answer, student_answer
            )
            res1, res2 = f1.result(), f2.result()

        final_grade, is_conflict = resolve_grade(
            res1["grade"], res2["grade"], self.conflict_limit, self.resolution
        )

        return {
            "student_id": student_id,
            "final_grade": final_grade,
            "conflict": is_conflict,
            "model_1": {"name": self.model_1, "grade": res1["grade"],
                        "reasoning": res1["raw_output"]},
            "model_2": {"name": self.model_2, "grade": res2["grade"],
                        "reasoning": res2["raw_output"]},
        }


def load_question_and_answer(filepath: str) -> Tuple[str, str]:
    r"""Load (question, answer) from a \question / \answer markdown file."""
    from .vision_grader import parse_answer_file

    with open(filepath, "r", encoding="utf-8") as f:
        return parse_answer_file(f.read())


def load_student_answers(folder: str) -> Dict[str, str]:
    """Load every student's answer text. Filename stem = student id."""
    students = {}
    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"Warning: answers folder '{folder}' does not exist.")
        return students

    for txt_file in sorted(folder_path.glob("*.txt")):
        students[txt_file.stem] = txt_file.read_text(encoding="utf-8")

    return students


def load_existing_results(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Load previously graded results so a run can resume."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {r["student_id"]: r for r in data.get("grades", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        print(f"Warning: could not read {filepath}; starting fresh.")
        return {}


def save_results(results: List[Dict[str, Any]], filepath: str):
    """Write results atomically so an interrupt cannot truncate the file."""
    output = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "total_students": len(results),
        "conflicts": sum(1 for r in results if r.get("conflict", False)),
        "grades": results,
    }
    tmp = f"{filepath}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp, filepath)


def grade_question(
    question_num: int = None,
    answers_file: str = None,
    students_folder: str = None,
    output_file: str = None,
    grader: HomeworkGrader = None,
):
    """
    Grade one question for every student.

    Reads student answers from questions/question_N/ (produced by the splitter)
    and the marking scheme from answers/correct_answer_N.md.
    """
    if answers_file:
        answer_path = Path(answers_file)
    elif question_num:
        answer_path = rubric_file(question_num, ANSWERS_FOLDER)
    else:
        print("Error: specify either question_num or answers_file")
        return

    if not answer_path.exists():
        print(f"Error: answer file '{answer_path}' not found!")
        return

    if students_folder is None:
        if question_num is None:
            print("Error: specify students_folder when using --file")
            return
        students_folder = str(Path(QUESTIONS_FOLDER) / f"question_{question_num}")

    output_file = output_file or (
        result_path(f"grades_question_{question_num}.json") if question_num else OUTPUT_FILE
    )

    print(f"Loading question and answer from {answer_path}...")
    question, correct_answer = load_question_and_answer(str(answer_path))
    print(f"Question: {question[:100]}..." if question else "No question section found.")

    print(f"Loading student answers from {students_folder}...")
    students = load_student_answers(students_folder)
    if not students:
        print(f"No student files found in '{students_folder}/'")
        return
    print(f"Found {len(students)} student submissions.")

    existing = load_existing_results(output_file)
    to_grade = {sid: a for sid, a in students.items() if sid not in existing}
    print(f"Already graded: {len(existing)}, New to grade: {len(to_grade)}")

    results = list(existing.values())
    grader = grader or HomeworkGrader()

    if to_grade:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    grader.grade_student, sid, question, correct_answer, ans
                ): sid
                for sid, ans in to_grade.items()
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                sid = futures[future]
                try:
                    results.append(future.result())
                    save_results(results, output_file)
                except Exception as exc:  # noqa: BLE001
                    print(f"Error grading student {sid}: {exc}")

    save_results(results, output_file)
    _print_summary(results, grader)
    print(f"\nResults saved to: {output_file}")


def _print_summary(results: List[Dict[str, Any]], grader: HomeworkGrader):
    print("\n--- Grading Summary ---")
    print(f"Total Students: {len(results)}")

    conflicts = [r for r in results if r.get("conflict", False)]
    print(f"Needs review (diff > {grader.conflict_limit} or a model failed): {len(conflicts)}")

    for c in conflicts[:10]:
        g1 = c.get("model_1", {}).get("grade")
        g2 = c.get("model_2", {}).get("grade")
        print(f"  {c['student_id']}: {grader.model_1}={g1}, {grader.model_2}={g2}, "
              f"final={c.get('final_grade')}")
    if len(conflicts) > 10:
        print(f"  ... and {len(conflicts) - 10} more")

    grades = [r["final_grade"] for r in results if r.get("final_grade") is not None]
    if grades:
        print(f"\nAverage Grade: {sum(grades) / len(grades):.1f}")
        print(f"Highest: {max(grades)}, Lowest: {min(grades)}")


def main():
    parser = argparse.ArgumentParser(description="Grade student homework from text")
    parser.add_argument("--question", "-q", type=int, help="Question number to grade")
    parser.add_argument("--file", "-f", help="Direct path to the answer file")
    parser.add_argument("--students-folder", "-s",
                        help="Folder of student answer .txt files "
                             "(default: questions/question_N)")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--model-1", default=MODEL_1_NAME)
    parser.add_argument("--model-2", default=MODEL_2_NAME)
    parser.add_argument("--conflict-limit", type=int, default=GRADE_CONFLICT_LIMIT)
    parser.add_argument("--resolution", choices=["max", "mean", "min"],
                        default=GRADE_RESOLUTION)
    args = parser.parse_args()

    if not args.question and not args.file:
        parser.error("specify --question N or --file PATH")

    grade_question(
        question_num=args.question,
        answers_file=args.file,
        students_folder=args.students_folder,
        output_file=args.output,
        grader=HomeworkGrader(
            model_1=args.model_1,
            model_2=args.model_2,
            conflict_limit=args.conflict_limit,
            resolution=args.resolution,
        ),
    )


if __name__ == "__main__":
    main()
