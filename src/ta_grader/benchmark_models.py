"""
Compare grading models on real submissions: grade, cost and latency.

Vision grading cost is dominated by how a model tokenizes page images, which
varies by an order of magnitude between families and is not predictable from
the per-token price alone. Measure before committing to a model.

    python benchmark_models.py --question 8 --limit 3
    python benchmark_models.py -q 8 --models google/gemini-2.5-flash,openai/gpt-5-nano

Costs a few cents. Results print as a table sorted by cost.
"""

import argparse
import concurrent.futures
import statistics
import time
from pathlib import Path

from .config import PDF_FOLDER, ANSWERS_FOLDER, get_client, student_id_for
from .grade_parsing import extract_single_grade
from .vision_grader import VisionGrader, get_grading_prompt_single, load_correct_answer

# Cheap vision-capable candidates, spread across model families so that a
# pair drawn from this list gives genuinely independent opinions.
# Per-request ceiling in seconds.
REQUEST_TIMEOUT = 120

DEFAULT_CANDIDATES = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-nano",
    "openai/gpt-4.1-nano",
    "qwen/qwen3-vl-32b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
    "amazon/nova-lite-v1",
]


def grade_once(client, model: str, system_prompt: str, images: list):
    """One grading call. Returns (grade, cost, seconds, prompt_tokens, error)."""
    content = [{"type": "text", "text": "Grade this student's homework submission:"}]
    for i, img in enumerate(images):
        content.append({"type": "text", "text": f"[Page {i + 1}]"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"}})

    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": content}],
            temperature=0,
            max_tokens=2000,
            # Some models stall indefinitely on a large image payload. Without
            # a per-call bound one bad model runs out the whole benchmark.
            timeout=REQUEST_TIMEOUT,
            extra_body={"usage": {"include": True}},
        )
    except Exception as exc:  # noqa: BLE001
        return None, 0.0, time.time() - started, 0, str(exc)[:80]

    elapsed = time.time() - started
    text = response.choices[0].message.content or ""
    usage = response.usage
    return (
        extract_single_grade(text),
        float(getattr(usage, "cost", 0) or 0),
        elapsed,
        getattr(usage, "prompt_tokens", 0),
        None,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark grading models")
    parser.add_argument("--question", "-q", type=int, required=True)
    parser.add_argument("--limit", "-l", type=int, default=3,
                        help="How many student PDFs to test on")
    parser.add_argument("--models", help="Comma-separated model ids to test")
    parser.add_argument("--out", help="Also append the summary table to this file")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Skip PDFs longer than this, to keep the run bounded")
    parser.add_argument("--pdf-folder", default=PDF_FOLDER)
    parser.add_argument("--answers-folder", default=ANSWERS_FOLDER)
    args = parser.parse_args()

    models = args.models.split(",") if args.models else DEFAULT_CANDIDATES

    correct_answer = load_correct_answer(args.answers_folder, args.question)
    if correct_answer is None:
        print(f"No answer file for question {args.question}")
        return
    system_prompt = get_grading_prompt_single(args.question, correct_answer)

    pdfs = sorted(p for p in Path(args.pdf_folder).iterdir()
                  if p.suffix.lower() == ".pdf")
    if args.max_pages:
        # A 17-page scan is a many-megabyte payload that dominates the run
        # without telling you more about model quality than a short one does.
        import fitz

        bounded = []
        for p in pdfs:
            try:
                doc = fitz.open(str(p))
                pages = len(doc)
                doc.close()
            except Exception:  # noqa: BLE001
                continue
            if pages <= args.max_pages:
                bounded.append(p)
        pdfs = bounded
    pdfs = pdfs[: args.limit]
    if not pdfs:
        print("No PDFs found")
        return

    grader = VisionGrader()
    print(f"Rendering {len(pdfs)} submissions...")
    rendered = {student_id_for(p): grader.pdf_to_images(str(p)) for p in pdfs}
    for sid, imgs in rendered.items():
        print(f"  {sid}: {len(imgs)} pages")

    # No retries: a stalling model should be reported as slow, not silently
    # retried three times past the per-request timeout.
    client = get_client().with_options(max_retries=0)
    results = {}

    for model in models:
        print(f"\nTesting {model} ...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(rendered)) as pool:
            futures = {
                pool.submit(grade_once, client, model, system_prompt, imgs): sid
                for sid, imgs in rendered.items()
            }
            per_student = {}
            for f in concurrent.futures.as_completed(futures):
                per_student[futures[f]] = f.result()

        grades = [r[0] for r in per_student.values() if r[0] is not None]
        cost = sum(r[1] for r in per_student.values())
        latency = statistics.mean(r[2] for r in per_student.values())
        tokens = statistics.mean(r[3] for r in per_student.values())
        errors = [r[4] for r in per_student.values() if r[4]]

        results[model] = {
            "grades": {sid: r[0] for sid, r in per_student.items()},
            "cost_per_student": cost / len(rendered),
            "latency": latency,
            "tokens": tokens,
            "errors": errors,
            "parsed": len(grades),
        }
        note = f" ERRORS: {errors[0]}" if errors else ""
        print(f"  grades={results[model]['grades']} "
              f"${results[model]['cost_per_student']:.5f}/student{note}")

    # --- Summary table ---
    print("\n" + "=" * 100)
    print(f"{'model':44}{'$/student':>11}{'prompt tok':>12}{'sec':>7}{'parsed':>8}  grades")
    print("-" * 100)
    for model, r in sorted(results.items(), key=lambda kv: kv[1]["cost_per_student"]):
        grades = ",".join(str(r["grades"][student_id_for(s)]) for s in pdfs)
        print(f"{model:44}{r['cost_per_student']:11.5f}{r['tokens']:12.0f}"
              f"{r['latency']:7.1f}{r['parsed']}/{len(pdfs):>7}  {grades}")

    print("\nPer 130-student class, one question, two models:")
    for model, r in sorted(results.items(), key=lambda kv: kv[1]["cost_per_student"]):
        print(f"  {model:44} ${r['cost_per_student'] * 130:7.2f}")


if __name__ == "__main__":
    main()
