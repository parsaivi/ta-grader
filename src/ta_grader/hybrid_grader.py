"""
Route each student to the grader that can actually read their submission.

Most submissions are typed and extract cleanly, so the cheap text route works.
A minority are handwritten or scanned: those extract to nothing, and the text
route would grade them as blank -- a silent zero for a student who did the work.

This module checks each PDF before grading and sends the unreadable ones to the
vision grader, then merges both sets into one result file. Every entry records
which route produced it.

    python hybrid_grader.py --question 8
    python hybrid_grader.py --question 8 --dry-run   # just show the routing
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from .config import (
    result_path,
    student_id_for,
    ANSWERS_FOLDER,
    MIN_ANSWERED_FRACTION,
    MIN_CHARS_PER_PAGE,
    NO_ANSWER_MARKER,
    PDF_FOLDER,
    QUESTIONS_FOLDER,
    VISION_GRADER_1,
    VISION_GRADER_2,
)

TEXT_ROUTE = "text"
VISION_ROUTE = "vision"


@dataclass
class Assessment:
    """Why a given student was routed the way they were."""
    student_id: str
    pdf_path: Path
    pages: int
    chars: int
    route: str
    reason: str

    @property
    def chars_per_page(self) -> float:
        return self.chars / self.pages if self.pages else 0.0


def assess_pdf(pdf_path: Path, min_chars_per_page: int = MIN_CHARS_PER_PAGE) -> Assessment:
    """Decide, from the PDF alone, whether text extraction is viable."""
    student_id = student_id_for(pdf_path)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        return Assessment(student_id, pdf_path, 0, 0, VISION_ROUTE,
                          f"PDF could not be opened ({exc})")

    try:
        pages = len(doc)
        chars = sum(len(page.get_text("text")) for page in doc)
    finally:
        doc.close()

    if pages == 0:
        return Assessment(student_id, pdf_path, 0, 0, VISION_ROUTE, "empty PDF")

    per_page = chars / pages
    if per_page < min_chars_per_page:
        return Assessment(
            student_id, pdf_path, pages, chars, VISION_ROUTE,
            f"only {per_page:.0f} chars/page -- handwritten or scanned",
        )

    return Assessment(student_id, pdf_path, pages, chars, TEXT_ROUTE,
                      f"{per_page:.0f} chars/page")


def reassess_after_split(
    assessments: List[Assessment],
    questions_folder: str,
    num_questions: int,
    min_answered_fraction: float = MIN_ANSWERED_FRACTION,
) -> List[Assessment]:
    """
    Re-route students whose text extracted but would not split.

    Text can extract fine and still be unusable -- answers living inside
    figures, or formatting the splitter cannot follow. Catching that here stops
    those students being graded as blank too.
    """
    if num_questions <= 0:
        return assessments

    updated = []
    for a in assessments:
        if a.route == VISION_ROUTE:
            updated.append(a)
            continue

        answered = 0
        found_any_file = False
        for q in range(1, num_questions + 1):
            path = Path(questions_folder) / f"question_{q}" / f"{a.student_id}.txt"
            if not path.exists():
                continue
            found_any_file = True
            body = path.read_text(encoding="utf-8").strip()
            if body and body != NO_ANSWER_MARKER:
                answered += 1

        # If the split step has not run yet there is nothing to judge.
        if not found_any_file:
            updated.append(a)
            continue

        fraction = answered / num_questions
        if fraction < min_answered_fraction:
            updated.append(Assessment(
                a.student_id, a.pdf_path, a.pages, a.chars, VISION_ROUTE,
                f"text extracted but splitter found only {answered}/{num_questions} answers",
            ))
        else:
            updated.append(a)

    return updated


def route_students(
    pdf_folder: str = PDF_FOLDER,
    questions_folder: str = QUESTIONS_FOLDER,
    num_questions: int = 0,
    min_chars_per_page: int = MIN_CHARS_PER_PAGE,
) -> List[Assessment]:
    """Assess every PDF and decide its route."""
    folder = Path(pdf_folder)
    if not folder.is_dir():
        print(f"Error: PDF folder '{pdf_folder}' not found.")
        return []

    pdfs = sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".pdf")
    assessments = [assess_pdf(p, min_chars_per_page) for p in pdfs]

    if num_questions:
        assessments = reassess_after_split(assessments, questions_folder, num_questions)

    return assessments


def print_routing(assessments: List[Assessment]):
    """Show the routing decision, so a surprise is visible before spending."""
    text = [a for a in assessments if a.route == TEXT_ROUTE]
    vision = [a for a in assessments if a.route == VISION_ROUTE]

    print(f"\nRouting {len(assessments)} submissions:")
    print(f"  text route   : {len(text)}")
    print(f"  vision route : {len(vision)}")

    if vision:
        print("\nGoing to the vision grader:")
        for a in vision:
            print(f"  {a.student_id:34} {a.reason}")


def _merge(results: List[Dict[str, Any]], route: str) -> List[Dict[str, Any]]:
    """Tag each result with the route that produced it."""
    for r in results:
        r["route"] = route
    return results


def grade_hybrid(
    question_num: int,
    pdf_folder: str = PDF_FOLDER,
    answers_folder: str = ANSWERS_FOLDER,
    questions_folder: str = QUESTIONS_FOLDER,
    output_file: str = None,
    num_questions: int = 0,
    dry_run: bool = False,
):
    """Grade one question, routing each student to a grader that can read them."""
    output_file = output_file or result_path(f"grades_question_{question_num}.json")

    assessments = route_students(pdf_folder, questions_folder, num_questions)
    if not assessments:
        print(f"No PDFs found in {pdf_folder}")
        return

    print_routing(assessments)

    if dry_run:
        print("\n(dry run -- nothing graded)")
        return

    text_ids = {a.student_id for a in assessments if a.route == TEXT_ROUTE}
    vision_assessments = [a for a in assessments if a.route == VISION_ROUTE]

    # Import lazily so --dry-run works without an API key.
    from .vision_grader import (
        VisionGrader,
        get_grading_prompt_single,
        load_correct_answer,
        load_existing_results,
        save_results,
    )

    correct_answer = load_correct_answer(answers_folder, question_num)
    if correct_answer is None:
        print(f"Error: {answers_folder}/correct_answer_{question_num}.md not found!")
        return

    existing = load_existing_results(output_file)
    results: List[Dict[str, Any]] = list(existing.values())
    already = set(existing.keys())

    # --- Text route ---
    text_todo = text_ids - already
    if text_todo:
        from .homework_grader import HomeworkGrader, load_question_and_answer

        answer_path = Path(answers_folder) / f"correct_answer_{question_num}.md"
        question, reference = load_question_and_answer(str(answer_path))

        q_folder = Path(questions_folder) / f"question_{question_num}"
        if not q_folder.exists():
            print(f"\nSkipping text route: {q_folder} does not exist "
                  f"(run the split step first).")
        else:
            print(f"\nText route: grading {len(text_todo)} students...")
            import concurrent.futures

            from .config import MAX_WORKERS
            from tqdm import tqdm

            grader = HomeworkGrader()
            answers = {}
            for sid in text_todo:
                path = q_folder / f"{sid}.txt"
                answers[sid] = path.read_text(encoding="utf-8") if path.exists() else ""

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {
                    pool.submit(grader.grade_student, sid, question, reference, ans): sid
                    for sid, ans in answers.items()
                }
                batch = []
                for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    try:
                        batch.append(f.result())
                    except Exception as exc:  # noqa: BLE001
                        print(f"Error grading {futures[f]}: {exc}")
                results.extend(_merge(batch, TEXT_ROUTE))
                save_results(results, output_file, question_num=question_num)

    # --- Vision route ---
    vision_todo = [a for a in vision_assessments if a.student_id not in already]
    if vision_todo:
        print(f"\nVision route: grading {len(vision_todo)} students...")
        import concurrent.futures

        from .config import MAX_WORKERS
        from tqdm import tqdm

        # Handwriting is the hardest input, so the vision route gets its own
        # model pair (same as the text pair by default).
        grader = VisionGrader(model_1=VISION_GRADER_1, model_2=VISION_GRADER_2)
        system_prompt = get_grading_prompt_single(question_num, correct_answer)

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(grader.grade_student_single_question,
                            str(a.pdf_path), system_prompt, question_num): a.student_id
                for a in vision_todo
            }
            batch = []
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    batch.append(f.result())
                except Exception as exc:  # noqa: BLE001
                    print(f"Error grading {futures[f]}: {exc}")
            results.extend(_merge(batch, VISION_ROUTE))

    save_results(results, output_file, question_num=question_num)

    # --- Summary ---
    by_route = {}
    for r in results:
        by_route.setdefault(r.get("route", "unknown"), []).append(r)

    print(f"\n--- Question {question_num} Summary ---")
    print(f"Total graded: {len(results)}")
    for route, items in sorted(by_route.items()):
        grades = [i["final_grade"] for i in items if i.get("final_grade") is not None]
        avg = f"{sum(grades) / len(grades):.1f}" if grades else "n/a"
        flagged = sum(1 for i in items if i.get("conflict"))
        print(f"  {route:7}: {len(items):3} students, avg {avg}, {flagged} need review")

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Grade with automatic text/vision routing per student"
    )
    parser.add_argument("--question", "-q", type=int, required=True)
    parser.add_argument("--num-questions", "-n", type=int, default=0,
                        help="Enables the post-split check for students whose "
                             "text extracted but would not split")
    parser.add_argument("--pdf-folder", default=PDF_FOLDER)
    parser.add_argument("--answers-folder", default=ANSWERS_FOLDER)
    parser.add_argument("--questions-folder", default=QUESTIONS_FOLDER)
    parser.add_argument("--output", "-o")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show routing without grading (no API calls)")
    args = parser.parse_args()

    grade_hybrid(
        question_num=args.question,
        pdf_folder=args.pdf_folder,
        answers_folder=args.answers_folder,
        questions_folder=args.questions_folder,
        output_file=args.output,
        num_questions=args.num_questions,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
