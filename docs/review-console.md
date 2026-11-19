# The review console

`ta-grader review` opens it in your browser, on localhost only.

The console turns a results file into a queue of decisions, one student at a
time, ordered by `uncertainty x marks_at_stake` rather than by student ID — so
the grade most likely to be wrong by the most marks is the first one you see.

The screen is two columns and a bar:

- **Left, pinned:** the whole submission as one scrolling document, opened on
  the page the model cited and ringed there in the accent colour. `←`/`→` move
  a page at a time; the scroll position survives a decision.
- **Docked under the page:** the verbatim quote the grade turns on, against the
  work it claims to quote. Its top border turns amber when the quote could not
  be found in the submission — the signature of a model reciting the answer key.
- **Right:** the flags, the model grades, and the paragraph of reasoning that
  concerns *this* part. Every part of the student is a chip in the header,
  coloured by its mark, so the state of the other parts is never hidden behind
  a tab.
- **Bottom, always reachable:** the decision bar. Quick picks fill the grade
  field rather than committing it, so a mis-click is not a decision. `Accept N`
  is one keystroke. `Withhold` records a deliberate null, never a zero.

Nothing needs the mouse:

| key | does | key | does |
|---|---|---|---|
| `J` `K` | part | `N` `P` | student |
| `0-9` | type a grade | `⏎` | accept the model's grade |
| `⌘⏎` | set the grade you typed | `⇧A` | accept every graded part on this student |
| `W` | withhold | `←` `→` | page |
| `F1`–`F6` | quick pick | `?` / `⌘K` | the full list |

Every shortcut is also a visible button — the keycaps in the decision bar are
how they are meant to be found.

**Cohort** is the whole class as a grid, and clicking any cell opens that part
in Triage. **Fairness** lists pairs of submissions whose evidence quotes are
near-identical but whose marks are not — grading errors by definition, needing
neither ground truth nor any trust in the model — and its student IDs are links
into Triage as well.

The theme is pinned in ``src/ta_grader/data/streamlit/config.toml``. Without it Streamlit follows
the browser's dark-mode setting and renders near-white text on the console's
paper-white surfaces, which makes the grade you just typed unreadable.

Every decision is appended to `grades_*.review.jsonl` with the model's original
grade beside yours. That ledger is the ground truth this project has never had:
it is the calibration set from the roadmap, accumulating over a term for free.

Marks are not final until a human signs off on the batch, and CSV export is
gated on that — assistant, not authority, as a step someone takes rather than a
sentence in a README.


## Rules the console must keep


`review_app.py` writes the same results file the grader writes, so every rule
above applies to it. The ones the version it replaces had already broken:

- **Accepting "the higher grade" where there is no grade must yield `null`.**
  The old `pick_higher` ended `... if present else 0`, so resolving a part
  neither model could grade wrote a zero against work nobody had looked at —
  ADR-003 violated through the review tool rather than through the grader.
- **The reviewer writes atomically and checks the file's mtime first.** A
  grading run may still be appending to it; the old reviewer held the whole file
  in session state and rewrote it on every save, which would delete any student
  a resumed run had added.
- **An override never overwrites the model's judgement, it records beside it.**
  Decisions append to `grades_*.review.jsonl` with the original grade, the new
  one, a reason and a timestamp. This is the only artifact here with value past
  the round: a growing set of human-corrected grades, which is the ground truth
  needed to measure accuracy at all.
- **A withheld grade exports as an empty cell, never as 0.** A spreadsheet is
  exactly where the null-vs-zero distinction gets destroyed.
- **Export is gated on batch sign-off.** No mark reaches a student without a
  human accepting the batch.


## Working on the console's layout


The console is a Streamlit app wearing a designed layout, and Streamlit fights
that in four specific ways. All four have already cost a day; read them before
touching `CSS` in `review_app.py`.

- **Emotion class selectors out-specify yours.** Streamlit styles blocks with
  two chained classes, so a plain `.st-key-foo {height: 40px}` loses. Layout
  declarations need `!important`.
- **`height` loses to `flex`.** Every block is `flex: 1 1 0%` inside a flex
  column, so a block in a parent with a height *grows to fill it* whatever
  height you set. A bar that must stay 40px needs `flex: 0 0 auto !important`
  on the block **and** on the `stLayoutWrapper` that Streamlit puts around it.
- **`stColumn:first-child` is a descendant selector.** Written without `>` it
  also matches the first column of every row nested inside that column, which
  is how the 40px page bar once inherited `height: calc(100vh - 110px)`. Scope
  chrome rules with `>` the whole way down.
- **Nested `st.columns` inherit the parent's height and wrap.** A row inside a
  row in a fixed-height bar lays out as a staircase of half-height columns.
  Keep bars flat: one row, one widget per column.

Two more that are not CSS:

- **`label {display: none}` hides toggles.** `st.toggle` renders its control
  *inside* the `<label>`. Hide labels per widget type, or pass
  `label_visibility="collapsed"`.
- **Keyboard shortcuts must bind in both documents.** The listener is injected
  by `components.html`, which is an iframe; a key press goes to whichever frame
  the browser has focused, and on a fresh load that is the iframe. `KEYS_JS`
  binds `keydown` on the parent document *and* on its own, and ignores
  `!e.isTrusted` so it never reacts to an event it dispatched itself.

Typed digits go through hidden buttons (`digit_0`..`digit_9`), not into the
number field. A value typed into a Streamlit widget is uncommitted until blur,
so anything that acts on it immediately reads the *previous* number. A button
click is the one interaction Streamlit cannot lose.

