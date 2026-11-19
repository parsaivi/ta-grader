# Adding a language

The grader needs to know four things about a language: the words students use
to start a question ("Question 3", "سوال ۳"), any digits other than 0-9, how
parts are labelled, and whether it is written right to left. A locale pack is
a TOML file holding exactly that.

1. Copy `src/ta_grader/data/locales/en.toml` to `<code>.toml`, where `<code>`
   is a short name such as `de` or `ar`. Put it in your workspace's `locales/`
   folder to try it, or in `src/ta_grader/data/locales/` to contribute it.
2. Fill in the fields:

   ```toml
   name = "German"
   rtl = false
   question_words = ["aufgabe", "frage"]
   digits = ""
   part_words = ["teil"]
   part_labels = []
   heading_examples = ["Aufgabe 1", "A1:", "1."]
   ```

   - `question_words` are matched case-insensitively at the start of a line,
     followed by a number.
   - `digits` lists a script's own digits in order 0 to 9. Several sets can
     follow each other (Persian ships two: Persian and Arabic-Indic).
   - `part_labels` are native labels in order; they map to a, b, c. Leave it
     empty if students letter parts a, b, c.
   - `rtl = true` turns on word-order repair for PDFs that store right-to-left
     lines backwards, and also recognises "1 Question" for the same reason.
3. Enable it in `course.toml`: `languages = ["de", "en"]`.
4. Check the splitter on one real submission before grading a class:

   ```bash
   ta-grader grade --route text --step extract
   python -m ta_grader.question_splitter --preview .work/extracted/<id>.txt -n 5
   ```

5. Check that your graders can read the language at all. Some cheap models
   return 0 for pages they cannot read. Grade three submissions you have marked
   yourself with `ta-grader benchmark -q 1 --limit 3` and compare.

If you contribute a pack, add a test in `tests/test_locales.py` with a short
real-looking heading sample.
