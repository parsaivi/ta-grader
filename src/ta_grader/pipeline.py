#!/usr/bin/env python3
"""
Homework Grading Pipeline

Text-based route (cheap, depends on extraction quality):
1. Extract text from PDFs (image descriptions via vision LLM)
2. Split each homework into individual question answers
3. Grade each question with dual-model verification

For handwritten or badly-extracting submissions, skip this and use
vision_grader.py, which sends the PDF pages straight to the models.

Usage:
    python pipeline.py --step all --pdf-folder students_pdf --num-questions 10 --question 8

Individual steps:
    python pipeline.py --step extract --pdf-folder students_pdf --smart-extract
    python pipeline.py --step split --num-questions 10
    python pipeline.py --step grade --question 8

Route each student automatically (text where it works, vision for handwriting):
    python pipeline.py --step hybrid --question 8 --num-questions 10
    python pipeline.py --step hybrid --question 8 --dry-run   # preview routing, free

Grade from a custom directory of text files:
    python pipeline.py --step grade-dir --answers-dir questions/question_1 \
        --correct-answer answers/correct_answer_1.md --output grades_q1.json
"""

import argparse

from .config import ANSWERS_FOLDER, EXTRACTED_FOLDER, PDF_FOLDER, QUESTIONS_FOLDER, result_path


def run_extraction(
    pdf_folder: str,
    output_folder: str,
    describe_images: bool = True,
    smart_extract: bool = False,
):
    """Step 1: Extract text from all PDFs."""
    print("=" * 50)
    print("STEP 1: Extracting text from PDFs")
    print("=" * 50)

    if smart_extract:
        from .smart_pdf_extractor import extract_all_pdfs as smart_extract_all

        print("Using SMART extraction (text + vision only for images)")
        smart_extract_all(pdf_folder, output_folder, describe_visuals=describe_images)
    else:
        from .pdf_extractor import extract_all_pdfs

        print("Using FULL vision extraction")
        extract_all_pdfs(pdf_folder, output_folder, describe_images=describe_images)

    print(f"\nExtracted files saved to: {output_folder}/")


def run_splitting(
    input_folder: str,
    output_folder: str,
    num_questions: int,
    use_llm: bool = True,
):
    """Step 2: Split homeworks into question-specific files."""
    from .question_splitter import split_all_homeworks

    print("=" * 50)
    print(f"STEP 2: Splitting homeworks into {num_questions} questions")
    print("=" * 50)

    split_all_homeworks(input_folder, output_folder, num_questions, use_llm=use_llm)
    print(f"\nSplit files saved to: {output_folder}/")


def run_grading(
    questions_folder: str,
    question_num: int,
    answers_folder: str,
    output_file: str = None,
):
    """Step 3: Grade a specific question for all students."""
    from .homework_grader import grade_question

    print("=" * 50)
    print(f"STEP 3: Grading Question {question_num}")
    print("=" * 50)

    grade_question(
        question_num=question_num,
        students_folder=f"{questions_folder}/question_{question_num}",
        output_file=output_file,
    )


def run_grading_from_dir(answers_dir: str, correct_answer_file: str, output_file: str):
    """Grade a question from an arbitrary directory of student answer files."""
    from .homework_grader import grade_question

    print("=" * 50)
    print(f"Grading from directory: {answers_dir}")
    print("=" * 50)

    grade_question(
        answers_file=correct_answer_file,
        students_folder=answers_dir,
        output_file=output_file,
    )


def run_full_pipeline(
    pdf_folder: str,
    num_questions: int,
    grade_question_num: int = None,
    describe_images: bool = True,
    smart_extract: bool = False,
    use_llm_splitter: bool = True,
):
    """Run extract -> split -> (optionally) grade."""
    run_extraction(pdf_folder, EXTRACTED_FOLDER, describe_images, smart_extract)
    run_splitting(EXTRACTED_FOLDER, QUESTIONS_FOLDER, num_questions, use_llm_splitter)

    if grade_question_num:
        # Route per student: handwritten submissions extract to nothing and
        # would otherwise be graded as blank.
        from .hybrid_grader import grade_hybrid

        grade_hybrid(
            question_num=grade_question_num,
            pdf_folder=pdf_folder,
            answers_folder=ANSWERS_FOLDER,
            questions_folder=QUESTIONS_FOLDER,
            num_questions=num_questions,
        )
    else:
        print("\n" + "=" * 50)
        print("Pipeline complete! To grade, run:")
        print("  python pipeline.py --step grade --question <N>")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Homework Grading Pipeline")

    parser.add_argument("--step",
                        choices=["extract", "split", "grade", "grade-dir", "hybrid", "all"],
                        default="all", help="Which step to run")
    parser.add_argument("--pdf-folder", default=PDF_FOLDER,
                        help="Folder containing student PDF files")
    parser.add_argument("--num-questions", type=int, default=10,
                        help="Number of questions in the homework")
    parser.add_argument("--question", type=int, help="Which question number to grade")
    parser.add_argument("--answers-dir", help="Directory of student answer .txt files "
                                              "(for grade-dir)")
    parser.add_argument("--correct-answer", help="Path to the correct answer file "
                                                 "(for grade-dir)")
    parser.add_argument("--answers-folder", default=ANSWERS_FOLDER,
                        help="Folder holding correct_answer_N.md files")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image description (faster, less accurate)")
    parser.add_argument("--smart-extract", action="store_true",
                        help="Use hybrid extraction (text + vision only for images)")
    parser.add_argument("--no-llm-split", action="store_true",
                        help="Split with regex heading detection only (no API calls)")
    parser.add_argument("--dry-run", action="store_true",
                        help="For the hybrid step: show routing without grading")

    args = parser.parse_args()
    describe_images = not args.no_images

    if args.step == "extract":
        run_extraction(args.pdf_folder, EXTRACTED_FOLDER, describe_images,
                       args.smart_extract)

    elif args.step == "split":
        run_splitting(EXTRACTED_FOLDER, QUESTIONS_FOLDER, args.num_questions,
                      use_llm=not args.no_llm_split)

    elif args.step == "grade":
        if not args.question:
            parser.error("--question is required for the grade step")
        run_grading(QUESTIONS_FOLDER, args.question, args.answers_folder, args.output)

    elif args.step == "hybrid":
        if not args.question:
            parser.error("--question is required for the hybrid step")
        from .hybrid_grader import grade_hybrid

        grade_hybrid(
            question_num=args.question,
            pdf_folder=args.pdf_folder,
            answers_folder=args.answers_folder,
            questions_folder=QUESTIONS_FOLDER,
            output_file=args.output,
            num_questions=args.num_questions,
            dry_run=args.dry_run,
        )

    elif args.step == "grade-dir":
        if not args.answers_dir:
            parser.error("--answers-dir is required for the grade-dir step")
        if not args.correct_answer:
            parser.error("--correct-answer is required for the grade-dir step")
        run_grading_from_dir(args.answers_dir, args.correct_answer,
                             args.output or result_path("grades_result.json"))

    else:  # all
        run_full_pipeline(
            args.pdf_folder,
            args.num_questions,
            args.question,
            describe_images,
            args.smart_extract,
            use_llm_splitter=not args.no_llm_split,
        )


if __name__ == "__main__":
    main()
