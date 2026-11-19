# Troubleshooting

**"No PDF files found"**
Check the files are in the workspace's `submissions/` folder and end in `.pdf` (case-insensitive).

**402 / "Insufficient credits" from the API**
The OpenRouter account is out of credit. Top up at
<https://openrouter.ai/credits>.

**404 "No endpoints found for model"**
The model id is wrong or no longer served. Check the current list at
<https://openrouter.ai/models> — ids are provider-prefixed, e.g.
`google/gemini-2.5-flash`, not `gemini-2.5-flash`.

**"LLM_KEY is not set"**
Put `LLM_KEY=...` in the workspace's `.env`.

**Everything comes back `[NO ANSWER PROVIDED]`**
Check `.work/extracted/` actually contains text. If it does, run
`python -m ta_grader.question_splitter --preview <file> -n N` to see where the splitter thinks
the boundaries are. If extraction is empty, the PDF is likely scanned images —
use `--route vision` instead of the text route.

**Grades look wrong**
Read both models' reasoning in the review UI. Usually the marking scheme is
underspecified; adding explicit point allocations to the rubric file (`rubrics/q<N>.md`)
helps a lot.

