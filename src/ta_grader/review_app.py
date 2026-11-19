#!/usr/bin/env python3
"""
The review console: where a human decides what the marks actually are.

    streamlit run review_app.py

This replaces `conflict_reviewer_app.py`, which was written for a shape of
results file the grader no longer produces. Four things were wrong with it and
each one is a design constraint here:

1. **It could not read a per-part file.** The only production round this project
   has ever run -- HW3, 81 students, three parts -- rendered as "Q1 Q2 Q3" with
   the part labels gone, `total_grade` absent, and a second grader column that
   was never called shown as if it had returned nothing.
2. **It sorted by student ID.** Which is to say it did not allocate attention at
   all: the 81st submission gets reviewed with the care left over after the
   first 80. That is the exact failure this project's README names as its
   reason for existing.
3. **It hid the page behind a tab.** Reviewing a grade means reading the
   student's work and the model's argument about it *at the same time*. Two
   tabs make that a memory exercise.
4. **It threw the human's judgement away.** An override overwrote the grade in
   place. The one artifact in this system with lasting value -- a human's
   corrected mark, next to the model's original -- was destroyed on every save.

What is here instead: a queue ordered by expected correction, one student at a
time, the whole submission beside the model's argument about it, and an
append-only ledger of every decision. The data layer lives in `review_store.py`
and is tested without a browser.

The layout is the one designed in `Human review panel redesign/` and its four
findings are the shape of this file:

- **Grading took too many clicks.** The decision form was at the bottom of a
  tab. It is now a bar fixed to the bottom of the viewport, never scrolled
  past, with `Accept N` as the cheapest action on screen.
- **The page and the argument were not really side by side.** The submission
  column is sticky and full height, and the evidence quote -- the one piece of
  text the grade turns on -- is docked underneath the page rather than filed
  away in the other column.
- **Parts were hidden behind `st.tabs`.** They are a chip strip in the student
  header now, always visible, coloured by grade.
- **No keyboard path.** Every action has a key, and every key has a visible
  button. See `KEY_HINTS`.

The submission renders as one scrollable document, not a page at a time. Paging
with `◀`/`▶` was a lie about how a reviewer reads a submission: they scan it,
land on the cited page, then look above and below it for the working.
"""

import glob
import html
import json
import os
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

import pymupdf as fitz
import streamlit as st
import streamlit.components.v1 as components

from ta_grader.config import RESULTS_FOLDER
from ta_grader.review_store import (
    FLAG_DISAGREEMENT,
    FLAG_EVIDENCE,
    FLAG_EXTREME,
    FLAG_NO_GRADE,
    FLAG_SINGLE_MODEL,
    FLAG_SPREAD,
    FLAG_TEXT,
    FLAG_VETO,
    VETO_TEXT,
    ReviewItem,
    ReviewSession,
    candidate_pdf_folders,
    find_pdf,
    suggestions,
)

st.set_page_config(
    page_title="Review console",
    page_icon="🧑‍⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --- The visual system --------------------------------------------------------
#
# One accent hue and one lightness/chroma pair per semantic role, so a colour
# never means two things. Taken verbatim from the reference design; the grade
# ramp in particular is load-bearing -- `None` is neutral grey, never a red,
# because a withheld mark is an absent one and not a low one.

CANVAS = "#f2f1ee"
SURFACE = "#ffffff"
SURFACE_ALT = "#fbfaf8"
SURFACE_SUNK = "#eceae5"
HAIRLINE = "#e4e2dc"
STRUCT = "#d9d6ce"
FIELD = "#ddd9d1"
INK = "#1c1b18"
INK_2 = "#57544d"
INK_3 = "#8a867c"
INK_4 = "#a8a49a"
ACCENT = "oklch(0.52 0.13 264)"
ACCENT_HOVER = "oklch(0.45 0.14 264)"
ACCENT_WASH = "oklch(0.96 0.03 264)"
ACCENT_RING = "oklch(0.94 0.03 264)"

SANS = "'IBM Plex Sans','IBM Plex Sans Arabic',system-ui,sans-serif"
MONO = "'IBM Plex Mono','IBM Plex Sans Arabic',ui-monospace,monospace"

RAIL_W = 78          # px, the icon rail
TOPBAR_H = 52        # px
HEADER_H = 58        # px, the sticky student header
SUBMISSION_FRAC = 0.575   # matches st.columns([1.15, 0.85])

# Only the hue changes between semantic roles -- same lightness, same chroma.
FLAG_HUE = {
    FLAG_VETO: 25,
    FLAG_NO_GRADE: 25,
    FLAG_EVIDENCE: 45,
    FLAG_DISAGREEMENT: 75,
    FLAG_SPREAD: 75,
    FLAG_EXTREME: 250,
    FLAG_SINGLE_MODEL: None,      # not a warning: an absence of cross-checking
}

REASONS = [
    "agreed with the model",
    "read the page -- model misread it",
    "model applied the rubric wrongly",
    "rubric does not cover this answer",
    "illegible -- graded by hand",
    "answer is not in the file",
    "arithmetic / partial credit adjustment",
    "other (see note)",
]

VIEWS = [
    ("Triage", "◎"),
    ("Cohort", "▦"),
    ("Fairness", "⇄"),
    ("Ledger", "≡"),
    ("Sign-off", "✓"),
]

KEY_HINTS: List[Tuple[str, str]] = [
    ("J / K", "part"),
    ("N / P", "student"),
    ("0-9", "type grade"),
    ("⏎", "accept"),
    ("⇧A", "accept all"),
    ("W", "withhold"),
    ("← →", "page"),
    ("?", "all keys"),
]

ALL_KEYS: List[Tuple[str, str]] = KEY_HINTS + [
    ("⌘⏎", "set the typed grade"),
    ("⏎", "in the grade field, sets what you typed"),
    ("F1-F6", "quick pick — fills, does not commit"),
    ("⌘K", "this panel"),
    ("Esc", "close"),
]


def grade_colour(grade: Optional[int], alpha: float = 1.0) -> str:
    """
    Red→green ramp in oklch, so 40 and 60 are as far apart perceptually as 80
    and 100 are. A withheld grade is neutral grey, never the colour of a zero.
    """
    if grade is None:
        return f"oklch(0.90 0.005 90 / {alpha})"
    t = max(0.0, min(1.0, grade / 100.0))
    return f"oklch({0.87 - 0.02 * t:.3f} {0.075 + 0.045 * t:.3f} {25 + 125 * t:.1f} / {alpha})"


def grade_ink(grade: Optional[int]) -> str:
    """Text colour that sits on `grade_colour` at the same hue."""
    if grade is None:
        return INK_3
    t = max(0.0, min(1.0, grade / 100.0))
    return f"oklch(0.40 0.11 {25 + 125 * t:.1f})"


def chip_style(hue: Optional[float]) -> str:
    if hue is None:
        return (
            f"display:inline-flex;align-items:center;gap:5px;padding:3px 9px;"
            f"border-radius:99px;font-size:11px;font-weight:600;background:{SURFACE_SUNK};"
            f"color:{INK_2};border:1px solid {FIELD};"
        )
    return (
        f"display:inline-flex;align-items:center;gap:5px;padding:3px 9px;"
        f"border-radius:99px;font-size:11px;font-weight:600;"
        f"background:oklch(0.96 0.03 {hue});color:oklch(0.42 0.14 {hue});"
        f"border:1px solid oklch(0.89 0.06 {hue});"
    )


CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ---- shell ------------------------------------------------------------- */
.stApp, body {background:$canvas; color:$ink; font-family:$sans; font-size:13.5px;}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, footer {display:none !important;}
header[data-testid="stHeader"] {height:0 !important; min-height:0 !important; background:transparent !important;}
/* The sidebar is setup, not the hot path -- its handle lives at the foot of
   the rail rather than competing with the top bar. */
[data-testid="stSidebarCollapsedControl"] {
  top:auto !important; bottom:12px !important; inset-inline-start:22px !important; z-index:200 !important;
}
.stMainBlockContainer {
  max-width:none !important;
  padding:$topbar_pad 0 0 $rail_pad !important;
}
/* The rail and the top bar are fixed, so their blocks contribute no height --
   but the main column's gap still counts them, which pushes the whole view
   down by the height of two rows that are not there. */
.stMainBlockContainer > [data-testid="stVerticalBlock"] {gap:0 !important;}
/* ...so the views that do want ordinary Streamlit spacing ask for it. */
.st-key-viewbody {
  gap:1rem !important; padding:20px 26px 80px 26px !important; max-width:1260px;
}
.stMain {background:$canvas;}
::selection {background:oklch(0.88 0.06 264);}
a {color:$accent; text-decoration:none;}
a:hover {color:$accent_hover; text-decoration:underline;}
@keyframes rise {from {opacity:0; transform:translateY(4px);} to {opacity:1; transform:none;}}

/* Streamlit wraps every block in a layout wrapper that grows to fill its
   parent and lets its column row wrap onto a second line. In a bar of a fixed
   height both behaviours are wrong: the row silently becomes as tall as the
   whole view and its controls end up hidden below the clip. The three bars --
   top, student header, decision -- opt out of both.
   Keep this rule above the per-bar blocks so they can still override it. */
.st-key-topbar [data-testid="stLayoutWrapper"],
.st-key-studenthdr [data-testid="stLayoutWrapper"],
.st-key-pagebar [data-testid="stLayoutWrapper"],
.st-key-decisionbar [data-testid="stLayoutWrapper"] {
  flex:0 0 auto !important; height:auto !important;
}
.st-key-topbar [data-testid="stHorizontalBlock"],
.st-key-studenthdr [data-testid="stHorizontalBlock"],
.st-key-pagebar [data-testid="stHorizontalBlock"] {
  flex-wrap:nowrap !important; height:auto !important; flex:0 0 auto !important;
}
.st-key-topbar [data-testid="stColumn"],
.st-key-studenthdr [data-testid="stColumn"],
.st-key-pagebar [data-testid="stColumn"],
.st-key-decisionbar [data-testid="stColumn"] {
  min-width:0 !important; height:auto !important;
}

/* Numbers are always mono and tabular -- a column of marks that shifts by a
   pixel per digit cannot be scanned. */
[data-testid="stMetricValue"], .num {font-family:$mono; font-variant-numeric:tabular-nums;}

/* ---- the icon rail ----------------------------------------------------- */
.st-key-rail {
  position:fixed; inset-inline-start:0; top:0; bottom:0; width:${rail}px; z-index:45;
  background:$surface_alt; border-inline-end:1px solid $struct;
  padding:10px 8px !important; gap:3px !important;
}
.st-key-rail > div {width:100%;}
.st-key-rail .stButton button {
  display:flex; flex-direction:column; align-items:center; gap:3px;
  padding:9px 4px; border-radius:9px; border:none; background:transparent;
  color:$ink_2; width:100%; min-height:0; line-height:1.25;
}
.st-key-rail .stButton button:hover {background:$surface_sunk; color:$ink;}
.st-key-rail .stButton button p {font-size:10px !important; font-weight:600;}

/* ---- the top bar ------------------------------------------------------- */
.st-key-topbar {
  position:fixed; top:0; inset-inline-start:${rail}px; inset-inline-end:0; height:${topbar}px;
  z-index:40; background:$surface_alt; border-bottom:1px solid $struct;
  padding:7px 18px 0 18px !important; overflow:hidden; gap:0 !important;
}
.st-key-topbar [data-testid="stHorizontalBlock"] {align-items:center !important; gap:14px !important;}
.st-key-topbar [data-testid="stVerticalBlock"] {gap:0 !important;}
.st-key-topbar [data-baseweb="select"] > div {
  min-height:30px; border-radius:7px; border-color:$field; background:$surface;
  font-family:$mono; font-size:12px;
}
/* Toggles live inside a <label>, so hiding labels wholesale hides the control.
   Only the two collapsed field labels are hidden, and they ask for it
   themselves via label_visibility. */
.st-key-topbar [data-testid="stCheckbox"] {margin:0;}
.st-key-topbar [data-testid="stCheckbox"] label {
  font-size:12px !important; color:$ink_2 !important; gap:7px;
}

/* ---- the sticky student header ----------------------------------------- */
.st-key-studenthdr {
  position:sticky; top:${topbar}px; z-index:30;
  background:$surface_alt; border-bottom:1px solid $hairline;
  padding:8px 20px !important; margin:0 !important; gap:0 !important;
  height:${header}px !important;
}
.st-key-studenthdr [data-testid="stHorizontalBlock"] {align-items:center !important; gap:8px !important;}
.st-key-studenthdr input {
  font-family:$mono !important; font-size:12.5px !important;
  border-radius:7px; background:$surface;
}

/* ---- triage split ------------------------------------------------------ */
/* The submission column is sticky and sized by its contents, not stretched to
   the viewport: Streamlit's blocks are flex children, and giving this column a
   fixed height makes them fight over the slack instead of stacking. */
/* Scoped with `>` all the way down on purpose: a descendant selector for
   `stColumn:first-child` also catches the first column of every row *inside*
   these columns, which is how the page bar's own columns ended up a full
   viewport tall. */
.st-key-triage_split > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
  align-items:flex-start !important; gap:0 !important;
}
.st-key-triage_split > [data-testid="stLayoutWrapper"]
  > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
  position:sticky !important; top:${split_top}px; align-self:flex-start !important;
  height:calc(100vh - ${split_top}px) !important; overflow:hidden;
  padding:0 !important; border-inline-end:1px solid $struct; background:$surface_sunk;
}
.st-key-triage_split > [data-testid="stLayoutWrapper"]
  > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child
  > [data-testid="stVerticalBlock"] {gap:0 !important; justify-content:flex-start !important;}
/* Streamlit's blocks are `flex: 1 1 0` and so grow to fill a sized parent,
   which beats any height set on them. The bar, the page and the quote each
   want their own fixed share, so they opt out of growing. */
.st-key-pagebar, .st-key-pdfview, .st-key-evbar,
.st-key-triage_split > [data-testid="stLayoutWrapper"]
  > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child
  > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {flex:0 0 auto !important;}
.st-key-triage_split > [data-testid="stLayoutWrapper"]
  > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
  padding:16px 20px 250px 20px !important; min-width:0;
}

/* The submission is one scrolling document. Its own iframe owns the scroll so
   the decision column beside it can move independently. */
.st-key-pdfview {
  height:calc(100vh - ${pdf_gap}px) !important; overflow-y:auto !important;
  padding:14px 16px 40px 16px !important; gap:16px !important;
  background:$surface_sunk; border:none !important;
}
/* No `scroll-behavior: smooth` here, and every scrollTo below asks for
   `behavior: 'instant'`. An animated scroll inside a Streamlit block is
   cancelled by the next re-render, and this container re-renders constantly --
   so a smooth scroll here is not a slower jump, it is no jump at all. */
.st-key-pdfview img {
  background:#fff; border:1px solid $struct; border-radius:3px;
  box-shadow:0 1px 3px rgba(28,27,24,.10);
}
.pglabel {display:flex; align-items:center; gap:8px; font-family:$mono;
          font-size:10.5px; color:$ink_3; text-transform:uppercase;
          letter-spacing:.08em; font-weight:600; margin-bottom:4px;}
.pgcited {text-transform:none; letter-spacing:0; color:oklch(0.45 0.12 264);
          background:$accent_wash; border:1px solid oklch(0.80 0.08 264);
          border-radius:5px; padding:1px 6px;}
/* The segmented zoom control, sized to sit inside a 40px bar. */
.st-key-zoom_pct [data-baseweb="button-group"] {gap:2px;}
.st-key-zoom_pct button {
  min-height:0 !important; height:26px !important; padding:0 9px !important;
  font-family:$mono; font-size:10.5px !important; border-radius:5px !important;
}
.st-key-pagebar {
  background:$surface_alt; border-bottom:1px solid $hairline;
  padding:5px 14px !important; gap:0 !important; height:40px !important; overflow:hidden;
}
.st-key-pagebar [data-testid="stHorizontalBlock"] {align-items:center !important; gap:6px !important;}
.st-key-evbar {padding:10px 14px !important; height:152px !important; overflow:auto;}

/* ---- the decision bar -------------------------------------------------- */
/* Fixed to the bottom of the viewport, spanning the decision column only, so
   the page it refers to is never covered. This is the main fix: the common
   action is one keystroke away from wherever the reviewer has scrolled to. */
.st-key-decisionbar {
  position:fixed; bottom:0;
  inset-inline-start:calc(${rail}px + (100vw - ${rail}px) * $subfrac);
  /* Streamlit's own block sets width:100%, which beats a left/right inset
     pair -- so the width is stated outright. */
  width:calc((100vw - ${rail}px) * $decfrac) !important;
  z-index:35; background:$surface_alt;
  border-top:1px solid $struct; border-inline-start:1px solid $struct;
  box-shadow:0 -8px 24px rgba(28,27,24,0.06);
  padding:8px 16px 4px 16px !important; gap:5px !important;
}
.st-key-decisionbar [data-testid="stHorizontalBlock"] {gap:6px !important; align-items:flex-end !important;}
.st-key-decisionbar [data-testid="stVerticalBlock"] {gap:4px !important;}
.st-key-decisionbar label {
  font-size:9.5px !important; text-transform:uppercase; letter-spacing:.07em;
  color:$ink_3 !important; font-weight:600;
}
.st-key-decisionbar .stButton button {min-height:36px; border-radius:8px;}
/* Streamlit hangs a -16px bottom margin on every markdown container, to cancel
   the trailing paragraph's own margin. A bare-<div> label has no such margin to
   cancel, so its box collapses to a few pixels while the text still occupies a
   full line -- and in a row aligned to flex-end the box, not the text, is what
   gets aligned. The QUICK label was drawn a line below the buttons it labels. */
.st-key-decisionbar [data-testid="stMarkdownContainer"] {margin-bottom:0 !important;}
/* A text label sitting in a row of 36px controls: same height, centred, so it
   lines up with the buttons whatever the row's alignment happens to be. */
.st-key-decisionbar .rowlabel {
  display:flex; align-items:center; height:36px;
  font-size:10px; text-transform:uppercase; letter-spacing:.08em;
  font-weight:700; color:$ink_3;
}
.st-key-draft_grade input {
  font-family:$mono !important; font-size:17px !important; font-weight:600;
  text-align:center; border-radius:8px;
}
.st-key-decisionbar [data-baseweb="select"] > div,
.st-key-decisionbar [data-baseweb="input"] {
  min-height:36px; border-radius:8px; border-color:$field; background:$surface; font-size:12.5px;
}

/* ---- buttons: quiet by default, one primary --------------------------- */
.stButton button, .stDownloadButton button {
  border-radius:7px; border:1px solid $field; background:$surface; color:$ink;
  font-weight:500; box-shadow:none; transition:none;
}
.stButton button:hover, .stDownloadButton button:hover {border-color:#b9b4a8; color:$ink;}
.stButton button[kind="primary"] {background:$accent; border-color:$accent; color:#fff;}
.stButton button[kind="primary"]:hover {background:$accent_hover; border-color:$accent_hover; color:#fff;}
.stButton button:disabled {background:$surface_alt; color:#c4c0b6; border-color:$hairline;}
.stButton button p {font-size:12.5px !important;}

/* ---- content atoms ---------------------------------------------------- */
.chips {display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px;}
.parthead {display:flex; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:12px;}
.parthead h2 {font-size:21px; font-weight:600; letter-spacing:-.015em; margin:0;}
.parthead .sub {font-size:11.5px; color:$ink_3; margin-top:1px;}
.prio {text-align:end; font-family:$mono; font-size:10.5px; color:$ink_4;
       line-height:1.6; white-space:nowrap;}
.ev {font-family:$mono; font-size:12.5px; line-height:1.75; color:$ink;
     max-height:96px; overflow:auto; white-space:pre-wrap; word-break:break-word;}
.evlabel {font-size:10.5px; text-transform:uppercase; letter-spacing:.08em;
          font-weight:700; color:$ink_2;}
.veto {display:flex; gap:10px; padding:11px 13px; border-radius:9px;
       background:oklch(0.96 0.025 25); border:1px solid oklch(0.88 0.06 25); margin-bottom:12px;}
.veto b {color:oklch(0.42 0.15 25);}
.decided {display:flex; align-items:center; gap:9px; padding:10px 13px; border-radius:9px;
          background:oklch(0.96 0.03 150); border:1px solid oklch(0.87 0.06 150);
          font-size:12.5px; margin-bottom:10px;}
.mcards {display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px;}
.mcard {flex:1; min-width:104px; padding:9px 12px; border-radius:9px;}
.mcard .name {font-size:10px; text-transform:uppercase; letter-spacing:.06em;
              color:$ink_3; font-weight:600; white-space:nowrap;}
.mcard .val {font-family:$mono; font-size:22px; font-weight:600; line-height:1.15;
             font-variant-numeric:tabular-nums;}
.kbd {font-family:$mono; font-size:10px; background:$surface; border:1px solid $field;
      border-bottom-width:2px; border-radius:4px; padding:1px 5px; color:#3a382f; font-weight:600;}
.keystrip {display:flex; align-items:center; gap:14px; flex-wrap:wrap;
           padding:6px 16px; margin:2px -16px -4px -16px;
           border-top:1px solid #eeece6; background:#f7f6f3;
           font-size:10.5px; color:$ink_3;}
.keystrip span.h {display:flex; align-items:center; gap:5px;}
.toast {position:fixed; bottom:150px; left:50%; transform:translateX(-50%); z-index:70;
        background:$ink; color:$surface_alt; padding:9px 16px; border-radius:99px;
        font-size:12.5px; font-weight:500; box-shadow:0 8px 24px rgba(28,27,24,.28);
        animation:rise .14s ease-out;}

/* ---- reasoning expanders --------------------------------------------- */
[data-testid="stExpander"] {border:1px solid $hairline; border-radius:9px; background:$surface;}
[data-testid="stExpander"] summary {padding:8px 12px; font-size:12.5px; font-weight:600;}
[data-testid="stExpander"] summary:hover {background:#faf9f6;}

/* ---- cohort + fairness ----------------------------------------------- */
h1.view {font-size:22px; font-weight:600; letter-spacing:-.02em; margin:0 0 4px 0;}
p.view {margin:0 0 16px 0; font-size:13px; color:$ink_2; max-width:680px; text-wrap:pretty;}
h2.view {font-size:15px; font-weight:600; margin:26px 0 3px 0;}
.legend {display:flex; gap:16px; align-items:center; margin-bottom:12px;
         flex-wrap:wrap; font-size:11.5px; color:$ink_2;}
.card {border:1px solid $struct; border-radius:11px; background:$surface; overflow:hidden;}
.grid {border-collapse:separate; border-spacing:0; width:100%;
       font-size:12.5px; font-variant-numeric:tabular-nums;}
.grid th {position:sticky; top:0; z-index:2; background:#f7f6f3; padding:9px 10px;
          font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:$ink_3;
          border-bottom:1px solid $struct; font-weight:700;}
.grid th.sid, .grid td.sid {text-align:start; padding-inline:14px; white-space:nowrap;}
.grid td {padding:3px; border-bottom:1px solid #f0eee8; text-align:center;}
.grid td.sid, .grid td.tot {font-family:$mono; font-size:12px;}
.grid a.cell {display:flex; align-items:center; justify-content:center; gap:3px;
              height:30px; border-radius:6px; font-family:$mono; font-size:12px;
              font-weight:500; text-decoration:none;}
.grid a.cell:hover {text-decoration:none; outline:2px solid $accent; outline-offset:-2px;}
.wrap {max-height:56vh; overflow:auto;}
.bars {display:flex; align-items:flex-end; gap:3px; height:70px;}
.bars i {flex:1; border-radius:3px 3px 0 0; min-height:2px; display:block;}
.axis {display:flex; justify-content:space-between; font-size:10px; color:$ink_4;
       font-family:$mono; margin-top:5px;}
.pair {display:grid; grid-template-columns:1fr 1fr;}
.pairhead {display:flex; align-items:center; gap:10px; padding:10px 14px;
           border-bottom:1px solid #f0eee8; background:#faf9f6; flex-wrap:wrap;}
</style>
""").substitute(
    canvas=CANVAS, surface=SURFACE, surface_alt=SURFACE_ALT, surface_sunk=SURFACE_SUNK,
    hairline=HAIRLINE, struct=STRUCT, field=FIELD, ink=INK, ink_2=INK_2, ink_3=INK_3,
    ink_4=INK_4, accent=ACCENT, accent_hover=ACCENT_HOVER, accent_wash=ACCENT_WASH,
    sans=SANS, mono=MONO,
    rail=RAIL_W, topbar=TOPBAR_H, rail_pad=f"{RAIL_W + 18}px", topbar_pad=f"{TOPBAR_H}px",
    header=HEADER_H, split_top=TOPBAR_H + HEADER_H,
    pdf_gap=TOPBAR_H + HEADER_H + 40 + 152,
    subfrac=SUBMISSION_FRAC, decfrac=round(1 - SUBMISSION_FRAC, 3),
)


# Every shortcut is mirrored by a visible button, and this listener does nothing
# but click those buttons -- so the keyboard path and the mouse path cannot
# drift apart. Streamlit has no key handling of its own; the listener is
# attached to the parent document from inside a component iframe.
KEYS_JS = Template("""
<script>
(function () {
  const doc = window.parent.document;
  if (doc.__rcKeys) doc.removeEventListener('keydown', doc.__rcKeys, true);

  // This listener lives on the parent document, so a key pressed while focus is
  // inside a component iframe never reaches it. On a fresh load focus *is* in
  // this iframe -- which is why, without this, no shortcut worked until the
  // reviewer had clicked somewhere first.
  try {
    const a = doc.activeElement;
    if (!a || a.tagName === 'IFRAME' || a === doc.documentElement) {
      if (a && a.blur) a.blur();
      doc.body.setAttribute('tabindex', '-1');
      doc.body.focus({ preventScroll: true });
    }
  } catch (err) {}

  const btn = (k) => doc.querySelector('.st-key-' + k + ' button');
  const click = (k) => { const b = btn(k); if (b && !b.disabled) { b.click(); return true; } return false; };
  const gradeBox = () => doc.querySelector('.st-key-draft_grade input');

  // Streamlit only commits a number field on blur, and that commit is a rerun
  // that would swallow a click queued behind it. So: park the intent on the
  // parent window (which survives reruns), blur to commit, and replay -- either
  // from the freshly injected copy of this script, or from the timer if the
  // value was unchanged and no rerun happened.
  //
  // Do not "commit" by dispatching a synthetic Enter: this listener would see
  // its own event and fire the Enter binding, which is Accept. That turned
  // every `⌘⏎ set 75` into `accept the model's grade`, on the wrong part.
  function commitThen(action) {
    window.parent.__rcPending = action;
    const el = gradeBox();
    if (el) el.blur();
    setTimeout(replay, 320);
  }
  function replay(tries) {
    const p = window.parent.__rcPending;
    if (!p) return;
    // The rerun that the blur causes re-renders this button, and the click has
    // to land after it exists -- so wait for it rather than assuming.
    if (!btn(p)) {
      if ((tries || 0) < 12) setTimeout(function () { replay((tries || 0) + 1); }, 120);
      return;
    }
    window.parent.__rcPending = null;
    click(p);
  }
  replay();

  // ← / → move a page at a time, which is all the old pager could do, and the
  // scroll container is in this document now so it is a scrollTo, not a rerun.
  const viewer = () => doc.querySelector('.st-key-pdfview');
  // Offsets are measured against the scroll container itself. `offsetTop` is
  // relative to whichever ancestor happens to be positioned, which in a
  // Streamlit block tree is not the one doing the scrolling.
  function offsetIn(v, el) {
    return (el.getBoundingClientRect().top - v.getBoundingClientRect().top) + v.scrollTop;
  }
  function pages(dir) {
    const v = viewer();
    if (!v) return;
    const marks = Array.prototype.slice.call(v.querySelectorAll('.pglabel'));
    const y = v.scrollTop;
    let target = null;
    if (dir > 0) { for (const m of marks) if (offsetIn(v, m) > y + 24) { target = m; break; } }
    else { for (const m of marks) if (offsetIn(v, m) < y - 24) target = m; }
    v.scrollTo({ top: target ? offsetIn(v, target) - 12 : (dir > 0 ? v.scrollHeight : 0),
                 behavior: 'instant' });
  }

  // Open on the page the model cited, but only when the item changes -- a rerun
  // caused by typing a digit must not yank the page out from under the reader.
  function openCited(tries) {
    const token = $cited_token;
    if (window.parent.__rcCited === token) return;
    const v = viewer();
    const label = v && v.querySelector('.pgcited') &&
                  v.querySelector('.pgcited').closest('.pglabel');
    // The pages are still arriving when this first runs, and until an image has
    // taken up its space every offset below it is wrong. So: keep measuring
    // until the number stops moving, then claim the token.
    if (!label) {
      if ((tries || 0) < 24) setTimeout(function () { openCited((tries || 0) + 1); }, 150);
      return;
    }
    const want = offsetIn(v, label) - 12;
    v.scrollTo({ top: want, behavior: 'instant' });
    if (Math.abs(v.scrollTop - want) > 2 && (tries || 0) < 24) {
      setTimeout(function () { openCited((tries || 0) + 1); }, 150);
      return;
    }
    if ((tries || 0) < 6) {   // let the remaining images settle before locking in
      setTimeout(function () { openCited((tries || 0) + 1); }, 200);
      return;
    }
    window.parent.__rcCited = token;
  }
  openCited();

  const handler = function (e) {
    if (!e.isTrusted) return;      // never react to an event this script made
    const t = doc.activeElement;
    const tag = t ? (t.tagName || '').toLowerCase() : '';
    const field = tag === 'input' || tag === 'textarea' || tag === 'select';
    const inGrade = !!(t && t.closest && t.closest('.st-key-draft_grade'));
    const cmd = e.metaKey || e.ctrlKey;

    if (cmd && e.key === 'Enter') { e.preventDefault(); commitThen('act_set'); return; }
    if (cmd && (e.key === 'k' || e.key === 'K')) { e.preventDefault(); click('keys_toggle'); return; }
    if (e.key === 'Escape') { if ($keys_open) click('keys_toggle'); if (t && t.blur) t.blur(); return; }
    if (cmd || e.altKey) return;

    // A search box has to be able to contain the letter n.
    if (field && !inGrade) { if (e.key === 'Enter') { /* let it commit */ } return; }

    if (/^F[1-6]$$/.test(e.key)) { e.preventDefault(); click('pick_' + (parseInt(e.key.slice(1), 10) - 1)); return; }

    // Digits go through real buttons rather than being poked into the number
    // field. Typing into the field leaves a value Streamlit has not committed,
    // and every scheme for committing it before acting on it is a race. A
    // button click is the one interaction Streamlit cannot lose.
    if (/^[0-9]$$/.test(e.key) && !inGrade) { e.preventDefault(); click('digit_' + e.key); return; }
    if (e.key === 'Backspace' && !field) { e.preventDefault(); click('digit_back'); return; }
    if (/^[0-9]$$/.test(e.key)) return;   // typed straight into the field: leave it alone
    // ⏎ accepts the model's grade -- except in the grade field, where it means
    // the number the reviewer just typed. Accepting and silently throwing that
    // number away is not a reading anyone would want.
    if (e.key === 'Enter') {
      e.preventDefault();
      if (inGrade) commitThen('act_set'); else click('act_accept');
      return;
    }
    if (inGrade) return;   // Backspace, arrows, everything else edits the number

    switch (e.key) {
      case 'j': case 'J': e.preventDefault(); click('part_next'); break;
      case 'k': e.preventDefault(); click('part_prev'); break;
      case 'K': e.preventDefault(); click('part_prev'); break;
      case 'n': e.preventDefault(); click('nav_next'); break;
      case 'p': e.preventDefault(); click('nav_prev'); break;
      case 'N': e.preventDefault(); click('nav_next'); break;
      case 'P': e.preventDefault(); click('nav_prev'); break;
      case 'w': case 'W': e.preventDefault(); commitThen('act_withhold'); break;
      case 'A': e.preventDefault(); click('act_accept_all'); break;
      case 'ArrowLeft': e.preventDefault(); pages(-1); break;
      case 'ArrowRight': e.preventDefault(); pages(1); break;
      case '?': e.preventDefault(); click('keys_toggle'); break;
    }
  };
  doc.__rcKeys = handler;
  doc.addEventListener('keydown', handler, true);
  // And on this iframe's own document. A key press goes to whichever frame the
  // browser considers focused, and on a fresh load that is this one -- moving
  // `activeElement` in the parent does not move the browser's frame focus, so
  // the only reliable fix is to listen in both places. The event only ever
  // fires in one of them, so there is no double handling.
  document.addEventListener('keydown', handler, true);
})();
</script>
""")


# --- Rendering the submission -------------------------------------------------

@st.cache_data(show_spinner=False)
def page_count(path: str, mtime: float) -> int:
    with fitz.open(path) as doc:
        return doc.page_count


@st.cache_data(show_spinner=False, max_entries=600)
def render_page(path: str, mtime: float, index: int, zoom: float) -> Optional[bytes]:
    """
    One page as PNG bytes, cached on (path, mtime, page, zoom).

    Per page rather than per document on purpose. The old reviewer rendered
    every page of every expanded submission on every widget interaction, and
    opened every conflicted student at once -- which on a round with a 19-in-20
    conflict rate meant rendering the entire class to answer one question.

    The whole-document view asks for every page of *one* student, which is the
    same cache used one entry at a time, so changing zoom on page 4 does not
    invalidate pages 1-3 at the old zoom.

    Bytes rather than base64 because `st.image` puts them in Streamlit's media
    store and hands the browser a URL. Inlined as data URIs, a six-page
    submission at 150% was a 4MB DOM that had to be re-sent on every rerun --
    and a rerun is what every keystroke in this console causes.
    """
    with fitz.open(path) as doc:
        if not 0 <= index < doc.page_count:
            return None
        pix = doc[index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")


ZOOM_STEPS = [100, 150, 200, 300]


def zoom_percent() -> float:
    """The current render zoom. 150% is legible for a phone photo of a page."""
    return float(st.session_state.get("zoom", 1.5))


def submission_pages(pdf: Path, mtime: float, count: int,
                     zoom: float, cited: List[int]) -> None:
    """
    The whole submission as one scrolling document.

    Paging with `◀`/`▶` was a lie about how a submission is read. A reviewer
    scans it, lands on the page the model cited, then looks above and below that
    page for the working the model did not quote -- and the old viewer made the
    second half of that a series of round-trips to the server.
    """
    with st.container(key="pdfview", height=760):
        for index in range(count):
            image = render_page(str(pdf), mtime, index, zoom)
            if image is None:
                continue
            number = index + 1
            is_cited = number in cited
            with st.container(key=f"pg{number}"):
                if is_cited:
                    # The cited page is ringed in the accent, and it is the one
                    # the view opens on.
                    css(
                        f".st-key-pg{number} img {{border-color:oklch(0.80 0.08 264) !important;"
                        f"box-shadow:0 0 0 3px {ACCENT_RING} !important;}}"
                    )
                st.markdown(
                    f'<div class="pglabel">page {number} / {count}'
                    + ('<span class="pgcited">model cited this page</span>' if is_cited else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                # PNG explicitly: left to itself `st.image` re-encodes to JPEG,
                # and JPEG ringing on a pencil scan is the difference between
                # reading a digit and guessing it.
                st.image(image, width="stretch", output_format="PNG")


# --- Small HTML helpers -------------------------------------------------------

def css(rules: str) -> None:
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


def button_css(key: str, decl: str, hover: str = "") -> None:
    """
    Paint one Streamlit button to the design's spec.

    Buttons carry state that CSS classes cannot -- a part chip is coloured by
    its grade, a rail item by whether it is the current view -- so the rule is
    generated next to the button rather than living in the stylesheet.
    """
    rule = f".st-key-{key} button {{{decl}}}"
    if hover:
        rule += f".st-key-{key} button:hover {{{hover}}}"
    rule += f".st-key-{key} button p {{margin:0; font-weight:inherit; font-size:inherit;}}"
    css(rule)


def flag_chips(item: ReviewItem, limit: int) -> str:
    flags = item.flags(limit)
    if not flags:
        return f'<span style="{chip_style(150)}">no flags</span>'
    return "".join(
        f'<span style="{chip_style(FLAG_HUE.get(f))}">{html.escape(FLAG_TEXT[f])}</span>'
        for f in flags
    )


def kbd(text: str) -> str:
    return f'<span class="kbd">{html.escape(text)}</span>'


# --- Session ------------------------------------------------------------------

def get_session(path: str, reviewer: str) -> ReviewSession:
    key = f"session::{path}"
    if key not in st.session_state:
        st.session_state[key] = ReviewSession(path, reviewer=reviewer)
    session: ReviewSession = st.session_state[key]
    session.reviewer = reviewer
    return session


def advance(session: ReviewSession, item: Optional[ReviewItem] = None,
            toast: str = "") -> None:
    """
    After a decision, move on.

    The cursor stays put rather than incrementing, because a decided item leaves
    the pending queue and the next one slides into its place. Incrementing here
    silently skips a grade -- which is the kind of bug that shows up as one
    unreviewed student in a batch nobody notices.

    Within a student, though, the cursor does move: to that student's next
    pending part, so a three-part submission is three keystrokes and not three
    round-trips through the queue.
    """
    if toast:
        st.session_state.toast = toast
    if item is not None:
        # Pin the student, not the queue position. Deciding their most valuable
        # part can promote a different student to the head of the queue, and
        # being yanked away from a submission you are halfway through reading is
        # how a part gets left undecided.
        siblings_pending = any(
            i for i in items_for_student(session, item.student_id) if not i.decided
        )
        st.session_state.pin = item.student_id if siblings_pending else None
        # Forwards from the part just decided, then round to any pending part
        # left behind it -- so a three-part student is three keystrokes in
        # reading order, and nothing is left undecided when it runs out.
        siblings = items_for_student(session, item.student_id)
        here = next((n for n, i in enumerate(siblings) if i.key == item.key), 0)
        ordered = siblings[here + 1:] + siblings[:here + 1]
        nxt = next((i for i in ordered if not i.decided), None)
        st.session_state.active_key = nxt.key if nxt else None
    st.session_state.pop("_draft_for", None)
    st.rerun()


# --- Views --------------------------------------------------------------------

def student_order(session: ReviewSession) -> List[str]:
    """Student IDs in queue priority order -- the student with the single most
    valuable pending item leads, so attention allocation still works even
    though the unit of navigation is now a student, not a part."""
    seen: List[str] = []
    seen_set = set()
    for item in session.queue():
        if item.student_id not in seen_set:
            seen_set.add(item.student_id)
            seen.append(item.student_id)
    return seen


def items_for_student(session: ReviewSession, student_id: str) -> List[ReviewItem]:
    return sorted(
        (i for i in session.items if i.student_id == student_id),
        key=lambda i: i.item_index,
    )


def all_student_ids(session: ReviewSession) -> List[str]:
    """Every student in the file, decided or not -- the search box's universe.
    The priority queue only ever surfaces students with something pending, so
    a fully-decided student is otherwise unreachable once graded."""
    seen: List[str] = []
    seen_set = set()
    for item in session.items:
        if item.student_id not in seen_set:
            seen_set.add(item.student_id)
            seen.append(item.student_id)
    return seen


def active_item(items: List[ReviewItem]) -> ReviewItem:
    """The part the decision column is about."""
    want = st.session_state.get("active_key")
    if want:
        found = next((i for i in items if i.key == want), None)
        if found is not None:
            return found
    pending = [i for i in items if not i.decided]
    return pending[0] if pending else items[0]


# --- Chrome -------------------------------------------------------------------

def render_rail(view: str, pending: int, pairs: int) -> None:
    with st.container(key="rail"):
        for name, glyph in VIEWS:
            key = f"rail_{name.lower().replace('-', '').replace(' ', '')}"
            on = name == view
            badge = {"Triage": pending, "Fairness": pairs}.get(name, 0)
            button_css(
                key,
                f"background:{ACCENT if on else 'transparent'};"
                f"color:{'#fff' if on else INK_2};",
                "" if on else f"background:{SURFACE_SUNK};",
            )
            label = f"{glyph}\n\n{name}" + (f" · {badge}" if badge else "")
            if st.button(label, key=key, width="stretch"):
                st.session_state.view = name
                st.rerun()
        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        button_css(
            "keys_toggle",
            f"border:1px dashed {STRUCT} !important; background:transparent !important;"
            f"color:{INK_3} !important; font-family:{MONO}; font-size:11px;",
        )
        if st.button("⌘K\n\nKeys", key="keys_toggle", width="stretch"):
            st.session_state.keys_open = not st.session_state.get("keys_open", False)
            st.rerun()


def metric_block(session: ReviewSession) -> str:
    total = len(session.items)
    done = sum(1 for i in session.items if i.decided)
    pct = round(100 * done / total) if total else 0
    stats = session.triage()
    blocking = stats["blocking"]

    def cell(value: str, label: str, colour: str = INK) -> str:
        return (
            f'<div style="padding:5px 12px;text-align:center;'
            f'border-inline-end:1px solid #efece5;">'
            f'<div style="font-family:{MONO};font-size:15px;font-weight:600;'
            f'font-variant-numeric:tabular-nums;line-height:1.2;color:{colour}">{value}</div>'
            f'<div style="font-size:9.5px;text-transform:uppercase;letter-spacing:.07em;'
            f'color:{INK_3};font-weight:600">{label}</div></div>'
        )

    return (
        f'<div style="display:flex;align-items:center;gap:14px">'
        f'<div style="flex:1;max-width:320px;min-width:110px">'
        f'<div style="display:flex;justify-content:space-between;font-size:11px;'
        f'color:#6f6c64;margin-bottom:4px;font-variant-numeric:tabular-nums">'
        f'<span>{done} of {total} decided</span>'
        f'<span style="font-family:{MONO}">{pct}%</span></div>'
        f'<div style="height:5px;border-radius:99px;background:{HAIRLINE};overflow:hidden">'
        f'<div style="height:100%;width:{pct}%;background:{ACCENT};border-radius:99px"></div>'
        f'</div></div>'
        f'<div style="display:flex;border:1px solid {HAIRLINE};border-radius:8px;'
        f'background:{SURFACE};overflow:hidden">'
        + cell(str(stats["pending"]), "pending")
        + cell(str(blocking), "blocking", "oklch(0.5 0.16 25)" if blocking else INK)
        + cell(f"{stats['expected_marks']:.1f}", "marks at stake")
        + "</div></div>"
    )


def render_topbar(session: ReviewSession, files: List[str]) -> str:
    with st.container(key="topbar"):
        c = st.columns([2.1, 2.3, 4.2, 2.4], gap="small")
        with c[0]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px">'
                f'<div style="width:22px;height:22px;border-radius:6px;background:{ACCENT};'
                f'display:flex;align-items:center;justify-content:center;color:#fff;'
                f'font-size:11px;font-weight:700">R</div>'
                f'<div style="font-weight:600;letter-spacing:-.01em">Review console</div></div>',
                unsafe_allow_html=True,
            )
        with c[1]:
            path = st.selectbox("Results file", files, key="results_file",
                                label_visibility="collapsed")
        with c[2]:
            st.markdown(metric_block(session), unsafe_allow_html=True)
        with c[3]:
            b = st.columns([3, 2])
            with b[0]:
                st.toggle(
                    "Blind mode", key="blind_mode",
                    help="Hide the model's grade until you have committed your own. "
                         "Slower per item, and the only way an override means anything "
                         "as evidence: a correction made after reading the model's "
                         "answer is anchored to it.",
                )
            with b[1]:
                fmt = {"parts": "per-part", "single": "single", "multi": "multi-question"}
                st.markdown(
                    f'<div style="font-size:11px;color:{INK_3};text-align:end;line-height:1.35">'
                    f'{fmt.get(session.format, session.format)}<br>'
                    f'{len(session.item_keys)} items · {len(all_student_ids(session))} students'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    return path


def render_keys_panel() -> None:
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:9px;font-size:12.5px;color:#3a382f">'
        f'{kbd(k)} {html.escape(label)}</div>'
        for k, label in ALL_KEYS
    )
    st.markdown(
        f'<div style="position:fixed;inset:0;z-index:60;background:rgba(28,27,24,.35);'
        f'display:flex;align-items:center;justify-content:center;padding:24px">'
        f'<div style="background:{SURFACE_ALT};border-radius:14px;padding:22px 24px;'
        f'width:520px;max-width:100%;box-shadow:0 24px 60px rgba(28,27,24,.25);'
        f'animation:rise .16s ease-out">'
        f'<div style="font-size:16px;font-weight:600;margin-bottom:14px">Keyboard shortcuts'
        f'</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 22px">{rows}</div>'
        f'<div style="margin-top:14px;font-size:11.5px;color:{INK_3}">'
        f'Press {kbd("Esc")} or {kbd("⌘K")} to close. Every shortcut is also a button — '
        f'the keycaps in the decision bar are the discovery path.</div></div></div>',
        unsafe_allow_html=True,
    )


# --- Triage -------------------------------------------------------------------

def view_triage(session: ReviewSession, pdf_folders: List[str]) -> None:
    limit = session.conflict_limit
    all_ids = all_student_ids(session)
    search = (st.session_state.get("student_search") or "").strip()

    if search:
        matches = [sid for sid in all_ids if search.lower() in sid.lower()]
        if not matches:
            student_header(session, None, limit, note=f"no id contains “{search}”")
            st.warning(f"No student id contains **{html.escape(search)}**.")
            return
        student_id = matches[0]
        students, cursor = matches, 0
    else:
        students = student_order(session)
        if not students:
            student_header(session, None, limit, note="queue empty")
            st.success("Queue empty — every item has been decided.")
            st.caption(
                "Sign-off and export are on the rail, or search above to revisit "
                "anyone already graded."
            )
            return
        pin = st.session_state.get("pin")
        if pin in students:
            cursor = students.index(pin)
        else:
            # The pinned student is clear (they have left the pending queue), so
            # the cursor -- which never incremented -- now points at whoever
            # slid into their place.
            cursor = min(st.session_state.get("cursor", 0), len(students) - 1)
            st.session_state.pin = students[cursor]
        st.session_state.cursor = cursor
        student_id = students[cursor]

    items = items_for_student(session, student_id)
    item = active_item(items)
    student_header(session, (students, cursor, items, item), limit)

    with st.container(key="triage_split"):
        left, right = st.columns([1.15, 0.85], gap=None)
        with left:
            show_submission(item, pdf_folders)
        with right:
            show_decision(session, item, limit)

    decision_bar(session, item, items, limit)


def student_header(session: ReviewSession, ctx, limit: int, note: str = "") -> None:
    """
    Who is on screen, what state each of their parts is in, and the two bulk
    moves. One flat row of columns -- the part chips are siblings of the student
    ID rather than a nested row, because a row inside a row is how this header
    grew to twice its height.
    """
    n_chips = 0 if ctx is None else len(ctx[2])
    searching = bool((st.session_state.get("student_search") or "").strip())
    with st.container(key="studenthdr"):
        cols = st.columns(
            [0.45, 0.45, 2.4] + [0.85] * n_chips + [0.9, 2.0, 0.5, 2.3],
            gap="small",
        )
        chip_at = 3
        total_at, search_at = 3 + n_chips, 4 + n_chips
        clear_at, accept_at = 5 + n_chips, 6 + n_chips

        # Searching replaces the queue, so there has to be a way back to it that
        # is not "select the text and delete it".
        if searching:
            button_css("search_clear", f"padding:4px 0;min-height:0;height:34px;color:{INK_3};")
            if cols[clear_at].button("✕", key="search_clear",
                                     help="clear the search and return to the queue"):
                st.session_state.student_search = ""
                st.session_state.active_key = None
                st.rerun()

        if ctx is None:
            cols[0].button("←", key="nav_prev", disabled=True)
            cols[1].button("→", key="nav_next", disabled=True)
            with cols[2]:
                st.markdown(
                    f'<div style="font-family:{MONO};font-size:15px;color:{INK_3};'
                    f'white-space:nowrap">{html.escape(note)}</div>',
                    unsafe_allow_html=True)
            with cols[search_at]:
                st.text_input("search", key="student_search",
                              placeholder="search any student id…",
                              label_visibility="collapsed")
            return

        students, cursor, items, item = ctx
        student_id = students[cursor]
        pending = [i for i in items if not i.decided]

        def go_to(index: int) -> None:
            st.session_state.cursor = index
            st.session_state.pin = students[index]
            st.session_state.active_key = None
            st.session_state.pop("_draft_for", None)
            st.rerun()

        if cols[0].button("←", key="nav_prev", disabled=cursor == 0,
                          help="previous student (P)"):
            go_to(cursor - 1)
        if cols[1].button("→", key="nav_next", disabled=cursor >= len(students) - 1,
                          help="next student (N)"):
            go_to(cursor + 1)

        with cols[2]:
            st.markdown(
                f'<div dir="auto" style="font-family:{MONO};font-size:19px;font-weight:600;'
                f'letter-spacing:-.01em;line-height:1.1">{html.escape(student_id)}</div>'
                f'<div style="font-size:11.5px;color:{INK_3};white-space:nowrap">'
                f'{cursor + 1} / {len(students)} · {len(pending)} of {len(items)} pending</div>',
                unsafe_allow_html=True,
            )

        # The part strip: every part's state visible while deciding any one of
        # them, which `st.tabs` made impossible.
        for i, part in enumerate(items):
            key = f"chip_{part.key}"
            on = part is item
            mark = "✓" if part.decided else ("!" if part.flags(limit) else "")
            text = "—" if part.grade is None else str(part.grade)
            button_css(
                key,
                f"background:{grade_colour(part.grade, 1 if on else .45)};"
                f"color:{grade_ink(part.grade)};font-family:{MONO};font-size:12px;"
                f"font-weight:600;padding:4px 6px;min-height:0;height:34px;"
                f"border:1.5px solid {ACCENT if on else 'transparent'} !important;"
                + (f"box-shadow:0 0 0 3px {ACCENT_RING};" if on else ""),
            )
            flags = part.flags(limit)
            if cols[chip_at + i].button(
                f"{part.key} {text}{mark}", key=key, width="stretch",
                help=", ".join(FLAG_TEXT[f] for f in flags) or "no flags",
            ):
                st.session_state.active_key = part.key
                st.session_state.pop("_draft_for", None)
                st.rerun()

        total = sum(i.grade for i in items if i.grade is not None)
        missing = sum(1 for i in items if i.grade is None)
        cols[total_at].markdown(
            f'<div style="font-family:{MONO};font-size:11.5px;color:{INK_3};'
            f'white-space:nowrap">tot {total}'
            + (f"<br>({missing} ungraded)" if missing else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        with cols[search_at]:
            st.text_input("search", key="student_search",
                          placeholder="search any student id…",
                          label_visibility="collapsed")

        # Accept-all: the common case on a student whose parts all look right.
        acceptable = [i for i in pending if i.grade is not None]
        with cols[accept_at]:
            button_css(
                "act_accept_all",
                f"background:{'oklch(0.96 0.03 150)' if acceptable else SURFACE_ALT};"
                f"border-color:{'oklch(0.85 0.06 150)' if acceptable else HAIRLINE} !important;"
                f"color:{'oklch(0.38 0.13 150)' if acceptable else '#c4c0b6'};font-weight:600;",
            )
            if st.button(f"✓ Accept {len(acceptable)} as-is  ⇧A", key="act_accept_all",
                         disabled=not acceptable, width="stretch",
                         help="Accept every part of this student that already has a "
                              "grade. Parts with no grade are never bulk-accepted — "
                              "there is nothing to confirm."):
                for part in acceptable:
                    session.accept(part, reason="accepted as graded (bulk, whole-student pass)")
                advance(session, toast=f"accepted {len(acceptable)} parts for {student_id}")

        # Part cursor for J/K. Hidden off-screen: the chips above are the visible
        # control, these only exist so the keys have something to click.
        css(".st-key-partnav {position:absolute; width:0; height:0; overflow:hidden;}")
        with st.container(key="partnav"):
            idx = items.index(item)
            if st.button("part_prev", key="part_prev"):
                st.session_state.active_key = items[max(0, idx - 1)].key
                st.rerun()
            if st.button("part_next", key="part_next"):
                st.session_state.active_key = items[min(len(items) - 1, idx + 1)].key
                st.rerun()


def show_submission(item: ReviewItem, pdf_folders: List[str]) -> None:
    pdf = find_pdf(item.student_id, pdf_folders)
    cited: List[int] = []
    mtime = count = 0
    if pdf is not None:
        # Read once and pass it down. Stat-ing the file twice in one render can
        # straddle a write and give the caption a different page count from the
        # document underneath it.
        mtime = pdf.stat().st_mtime
        count = page_count(str(pdf), mtime)
        cited = [p for p in item.pages() if 1 <= p <= count]
    # What the viewer should open on. The token changes only when the item or its
    # cited pages do, which is the signal the scroll script waits for.
    st.session_state.cited_token = f"{item.student_id}:{item.key}:{cited}"

    # Two columns and one widget each. Four zoom buttons in nested columns is a
    # row inside a row, and Streamlit lays those out by giving each nested
    # column a share of the parent's height -- which turns a 40px bar into a
    # 600px staircase.
    with st.container(key="pagebar"):
        bar = st.columns([6.5, 3.5], gap="small")
        with bar[0]:
            if pdf is None:
                text = "no PDF found"
            elif cited:
                text = (f"{count} pages · model cited page"
                        f"{'s' if len(cited) > 1 else ''} {', '.join(map(str, cited))}")
            else:
                text = f"{count} pages · the model did not say which page"
            st.markdown(
                f'<div style="font-family:{MONO};font-size:11.5px;color:{INK_2};'
                f'padding-top:7px">{html.escape(text)}'
                f'<span style="color:{INK_4}"> · {html.escape("← →")} page</span></div>',
                unsafe_allow_html=True,
            )
        with bar[1]:
            percent = bar[1].segmented_control(
                "zoom", ZOOM_STEPS, default=int(zoom_percent() * 100),
                key="zoom_pct", label_visibility="collapsed",
                help="How large the pages are rendered. Higher is slower and "
                     "costs memory; 150% is legible for most scans.",
            )
            if percent and percent != int(zoom_percent() * 100):
                st.session_state.zoom = percent / 100
                st.rerun()

    if pdf is None:
        st.warning(
            f"No PDF found for **{item.student_id}** in "
            f"{', '.join(pdf_folders) or 'any students_pdf* folder'}."
        )
        st.caption(
            "The student ID is the filename stem. A submission staged as "
            "`UNKNOWN_…` keeps that name until it is renamed."
        )
    else:
        with st.spinner("Rendering submission…"):
            submission_pages(pdf, mtime, count, zoom_percent(), cited)

    # The evidence quote is docked against the page, not filed in the decision
    # column. It is the one piece of text the grade turns on, and reading it
    # means reading it *next to* the handwriting it claims to quote.
    amber = item.evidence_ok is False
    with st.container(key="evbar"):
        css(
            f".st-key-evbar {{background:{SURFACE_ALT};"
            f"border-top:2px solid {'oklch(0.75 0.13 45)' if amber else STRUCT};}}"
        )
        if not item.evidence:
            st.markdown(
                f'<div style="font-size:11.5px;color:{INK_3}">No evidence quote recorded '
                f'for this part — the grade rests on the reasoning alone.</div>',
                unsafe_allow_html=True,
            )
            return
        verdict, hue = {
            True: ("quote checks out", 150),
            False: ("quote could not be matched in the submission", 45),
            None: ("not checked", None),
        }[item.evidence_ok]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
            f'<span class="evlabel">Evidence the grade turns on</span>'
            f'<span style="{chip_style(hue)}">{html.escape(verdict)}</span></div>'
            f'<div dir="auto" class="ev">{html.escape(item.evidence)}</div>',
            unsafe_allow_html=True,
        )
        if amber:
            st.markdown(
                '<div style="font-size:11.5px;color:oklch(0.42 0.14 45);margin-top:6px">'
                'A quote that is not in the student’s work is the signature of a model '
                'reciting the answer key. Treat this grade as unsupported until you '
                'have read the page.</div>',
                unsafe_allow_html=True,
            )


def blind_key(item: ReviewItem) -> str:
    return f"blind::{item.student_id}::{item.key}"


def show_blind(session: ReviewSession, item: ReviewItem) -> None:
    """
    Mark it yourself first, then see what the model said.

    The console's default layout has a flaw that is invisible until you want to
    use its output as evidence: it shows the model's grade before the human
    decides. Every override is then anchored to that number, so the corrections
    it produces cannot be used to measure how accurate the model is -- they were
    made by someone who had already read its answer.

    Doc 11 Stage 0 asks for 30 submissions hand-graded *before* looking at the
    model's grades, as a one-week project. This is that project, running inside
    the work the TA had to do anyway: mark blind, commit, then reveal. The
    ledger records both numbers, and the difference between them is the accuracy
    figure this project has never had and which everything downstream waits on.
    """
    st.markdown(
        f'<div style="padding:13px 14px;border-radius:10px;'
        f'border:1px solid oklch(0.86 0.05 264);background:{ACCENT_WASH};margin-bottom:14px">'
        f'<div style="font-weight:600;margin-bottom:3px">Blind mode — mark it yourself first</div>'
        f'<div style="font-size:12px;color:{INK_2}">The model’s grade is hidden until you '
        f'commit. Both numbers go to the ledger; the difference between them is the only '
        f'honest accuracy figure.</div></div>',
        unsafe_allow_html=True,
    )
    with st.form(key=f"blind_{item.student_id}_{item.key}"):
        c1, c2 = st.columns([1, 3])
        grade = c1.number_input("Your grade", min_value=0, max_value=100, step=1, value=0)
        note = c2.text_input("One line of reasoning", placeholder="why that mark")
        b1, b2, _ = st.columns([2, 2, 3])
        commit = b1.form_submit_button("Commit and reveal", type="primary", width="stretch")
        cannot = b2.form_submit_button("Can’t grade this", width="stretch")
    if commit:
        st.session_state[blind_key(item)] = {"grade": int(grade), "note": note}
        st.rerun()
    if cannot:
        st.session_state[blind_key(item)] = {"grade": None, "note": note or "not gradeable"}
        st.rerun()


def show_decision(session: ReviewSession, item: ReviewItem, limit: int) -> None:
    st.markdown(f'<div class="chips">{flag_chips(item, limit)}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="parthead"><div>'
        f'<h2>{html.escape(item.label)}</h2>'
        f'<div class="sub">worth {item.weight:g} marks · '
        f'{("tag " + html.escape(item.tag) + " — never affects the mark") if item.tag else "no label recorded"}'
        f'</div></div>'
        f'<div class="prio"><div>priority {item.priority(limit):.2f}</div>'
        f'<div>uncertainty {item.uncertainty(limit):.2f} × swing {item.swing():.0f}pts '
        f'× {item.weight:g}</div></div></div>',
        unsafe_allow_html=True,
    )

    if item.veto:
        st.markdown(
            f'<div class="veto"><span style="font-size:15px">⚠</span><div>'
            f'<b>Grade withheld — {html.escape(VETO_TEXT.get(item.veto, item.veto))}</b>'
            f'<div style="font-size:12px;color:{INK_2};margin-top:2px">This is not a low '
            f'mark, it is an absent one. Read the page and set the grade yourself, or '
            f'record that it cannot be graded.</div></div></div>',
            unsafe_allow_html=True,
        )

    blind = st.session_state.get("blind_mode", False)
    committed = st.session_state.get(blind_key(item))
    if blind and committed is None:
        show_blind(session, item)
        return

    if committed is not None:
        mine, theirs = committed["grade"], item.grade
        delta = ""
        if mine is not None and theirs is not None:
            delta = f'<span style="color:{INK_2}"> · {mine - theirs:+d} vs the model’s {theirs}</span>'
        st.markdown(
            f'<div class="decided" style="background:{ACCENT_WASH};'
            f'border-color:oklch(0.86 0.05 264)">'
            f'<span>◉</span><div><b>Your blind grade — {"—" if mine is None else mine}</b>'
            f'{delta}<div style="font-size:11.5px;color:{INK_2}">Both numbers go into the '
            f'ledger whichever you settle on.</div></div></div>',
            unsafe_allow_html=True,
        )

    if item.decided and item.decision:
        d = item.decision
        shown = "—" if d.get("grade") is None else d["grade"]
        c = st.columns([7, 1.6])
        with c[0]:
            st.markdown(
                f'<div class="decided"><span style="color:oklch(0.45 0.13 150)">✓</span>'
                f'<div><b>Decided — {shown}</b>'
                f'<span style="color:{INK_2}"> · {html.escape(str(d.get("reason") or ""))}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with c[1]:
            if st.button("Undo", key=f"undo_{item.student_id}_{item.key}", width="stretch"):
                session.undo(item)
                advance(session, toast=f"reopened {item.label}")

    cards = []
    for name, grade in item.model_grades.items():
        cards.append((name.split("/")[-1], grade))
    cards.append(("current", item.grade))
    st.markdown(
        '<div class="mcards">'
        + "".join(
            f'<div class="mcard" style="border:1.5px solid {grade_ink(g)};'
            f'background:{grade_colour(g, .28)}">'
            f'<div class="name">{html.escape(n)}</div>'
            f'<div class="val" style="color:{grade_ink(g)}">{"—" if g is None else g}</div></div>'
            for n, g in cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    meta = []
    if item.confidence:
        meta.append(f"confidence {item.confidence}")
    if item.spread is not None:
        meta.append(f"spread {item.spread}")
    if item.samples:
        meta.append("samples " + ", ".join("—" if s is None else str(s) for s in item.samples))
    if item.legibility:
        meta.append(f"legibility {item.legibility}")
    if item.presence:
        meta.append(f"presence {item.presence}")
    if item.single_model:
        meta.append("single-model run — nothing cross-checked")
    if meta:
        st.markdown(
            '<div class="chips">'
            + "".join(
                f'<span style="display:inline-flex;padding:3px 9px;border-radius:6px;'
                f'font-size:11px;font-family:{MONO};background:{SURFACE_SUNK};'
                f'color:{INK_2};border:1px solid {HAIRLINE}">{html.escape(m)}</span>'
                for m in meta
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    for model in item.reasoning:
        text, sliced = item.reasoning_for(model)
        scope = f"part ({item.key})" if sliced else "full reasoning"
        words = len(text.split()) if text else 0
        with st.expander(f"{model.split('/')[-1]} · {scope} · {words}w", expanded=False):
            st.markdown(
                f'<div dir="auto" style="font-size:13px;line-height:1.65;color:#3a382f;'
                f'white-space:pre-wrap">{html.escape(text) if text else "<em>[no reasoning recorded]</em>"}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if sliced:
                with st.expander("all parts", expanded=False):
                    st.markdown(
                        f'<div dir="auto" style="font-size:12.5px;line-height:1.6;'
                        f'white-space:pre-wrap">{html.escape(item.reasoning[model] or "[none]")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def quick_picks(item: ReviewItem) -> List[Tuple[str, int]]:
    """
    The suggestions `review_store` computes, minus the null one.

    `withhold -- no grade` is a button of its own in the bar, and a quick pick
    that silently means "no mark" next to five that mean numbers is the shape of
    a mis-click that ends up in an export.
    """
    short = {
        "median of": "median",
        "higher (": "higher ",
        "lower (": "lower ",
        "midpoint (": "mid ",
        "the one grade given (": "only ",
    }
    picks: List[Tuple[str, int]] = []
    for label, value in suggestions(item):
        if value is None:
            continue
        # The bar is narrow and these are read at a glance, so "higher (75)"
        # becomes "higher 75". The number is the part that matters.
        for prefix, replacement in short.items():
            if label.startswith(prefix):
                label = f"{replacement}{value}"
                break
        picks.append((label.rstrip(")"), value))
    have = {v for _, v in picks}
    for edge in (0, 100):
        if edge not in have and len(picks) < 6:
            picks.append((str(edge), edge))
    return picks[:6]


def keypad() -> None:
    """
    Eleven off-screen buttons: the digits and a backspace.

    `0-9` in the design types a grade. The obvious implementation -- have the
    key listener write into the number field -- leaves a value Streamlit has
    not committed, so the very next action reads the *old* number. Typing
    through buttons keeps one source of truth: the server's.
    """
    css(".st-key-keypad {position:absolute; width:0; height:0; overflow:hidden;}")
    with st.container(key="keypad"):
        for digit in range(10):
            if st.button(str(digit), key=f"digit_{digit}"):
                typed = (st.session_state.get("typed") or "") + str(digit)
                # Third digit only makes sense as 100; anything else rolls over
                # so a mistyped "857" does not silently become 85.
                if len(typed) > 3 or int(typed) > 100:
                    typed = str(digit)
                st.session_state.typed = typed
                st.session_state.draft_grade = int(typed)
                st.rerun()
        if st.button("back", key="digit_back"):
            typed = (st.session_state.get("typed") or "")[:-1]
            st.session_state.typed = typed
            st.session_state.draft_grade = int(typed) if typed else 0
            st.rerun()


def decision_bar(session: ReviewSession, item: ReviewItem,
                 items: List[ReviewItem], limit: int) -> None:
    """
    The bar fixed to the bottom of the viewport.

    Reaching the old form meant scrolling past two collapsed expanders; using it
    meant number → reason → button, per part, per student. Here the whole
    decision is on screen at all times and the common case -- the model was
    right -- is one keystroke.
    """
    committed = st.session_state.get(blind_key(item))
    default = committed["grade"] if committed else item.grade
    stamp = (item.student_id, item.key)
    if st.session_state.get("_draft_for") != stamp:
        st.session_state["_draft_for"] = stamp
        st.session_state["draft_grade"] = int(default) if default is not None else 0
        st.session_state["decision_note"] = committed["note"] if committed else ""
        st.session_state["typed"] = ""

    keypad()

    with st.container(key="decisionbar"):
        picks = quick_picks(item)
        row = st.columns([0.8] + [1] * len(picks) + [max(1, 6 - len(picks))], gap="small")
        row[0].markdown('<div class="rowlabel">Quick</div>',
                        unsafe_allow_html=True)
        for i, (label, value) in enumerate(picks):
            key = f"pick_{i}"
            button_css(key, "font-size:11.5px;padding:3px 8px;min-height:0;")
            if row[i + 1].button(f"{label}  F{i + 1}", key=key, width="stretch",
                                 help="fills the grade field — it does not commit"):
                # Filling rather than committing: a mis-clicked suggestion should
                # not be a decision.
                st.session_state["draft_grade"] = int(value)
                st.session_state["typed"] = str(int(value))
                st.rerun()

        c = st.columns([1.15, 3.1, 3.6], gap="small")
        with c[0]:
            st.number_input("Grade", min_value=0, max_value=100, step=1,
                            key="draft_grade")
        with c[1]:
            reason = st.selectbox("Reason", REASONS, key="decision_reason")
        with c[2]:
            note = st.text_input("Note", key="decision_note",
                                 placeholder="what you saw on the page")

        extra = {}
        if committed is not None:
            extra = {"blind": True, "blind_grade": committed["grade"],
                     "model_grade": item.grade}

        act = st.columns([2.2, 2.2, 1.8, 1.6], gap="small")
        with act[0]:
            if st.button("Set grade  ⌘⏎", key="act_set", type="primary", width="stretch"):
                session.apply(item, st.session_state["draft_grade"],
                              reason=reason, note=note, extra=extra)
                advance(session, item,
                        f"{item.label} → {st.session_state['draft_grade']}")
        with act[1]:
            button_css(
                "act_accept",
                f"background:{grade_colour(item.grade, .5)};color:{grade_ink(item.grade)};"
                f"border-color:{grade_ink(item.grade) if item.grade is not None else FIELD} !important;"
                f"font-weight:600;",
            )
            label = "Accept —" if item.grade is None else f"Accept {item.grade}"
            if st.button(f"{label}  ⏎", key="act_accept", width="stretch",
                         disabled=item.grade is None):
                session.apply(item, item.grade, reason="accepted as graded", extra=extra)
                advance(session, item, f"{item.label} accepted at {item.grade}")
        with act[2]:
            button_css(
                "act_withhold",
                "border-color:oklch(0.86 0.06 25) !important;color:oklch(0.45 0.15 25);",
                "background:oklch(0.97 0.02 25);",
            )
            if st.button("Withhold  W", key="act_withhold", width="stretch",
                         help="A deliberate null. The export leaves the cell empty "
                              "rather than writing a zero the student did not earn."):
                session.apply(item, None, reason=reason, note=note or "not gradeable",
                              extra=extra)
                advance(session, item, f"{item.label} withheld")

        if st.session_state.get("show_keys_strip", True):
            st.markdown(
                '<div class="keystrip">'
                + "".join(
                    f'<span class="h">{kbd(k)} {html.escape(label)}</span>'
                    for k, label in KEY_HINTS
                )
                + "</div>",
                unsafe_allow_html=True,
            )


# --- Cohort -------------------------------------------------------------------

def view_cohort(session: ReviewSession) -> None:
    limit = session.conflict_limit
    keys = session.item_keys
    by_student: Dict[str, Dict[str, ReviewItem]] = {}
    for item in session.items:
        by_student.setdefault(item.student_id, {})[item.key] = item

    st.markdown(
        '<h1 class="view">The cohort at a glance</h1>'
        '<p class="view">Colour is the mark. Grey is a withheld or missing grade — the '
        'one cell colour that does not mean the student did badly. Click any cell to '
        'open that part in triage.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="legend">'
        f'<span style="display:flex;align-items:center;gap:7px">0'
        f'<span style="width:120px;height:9px;border-radius:99px;background:linear-gradient('
        f'90deg,{grade_colour(0)},{grade_colour(50)},{grade_colour(100)})"></span>100</span>'
        f'<span style="display:flex;align-items:center;gap:6px">'
        f'<span style="width:14px;height:14px;border-radius:3px;background:{grade_colour(None)};'
        f'border:1px solid #d3d0c8"></span> withheld / no grade</span>'
        f'<span style="display:flex;align-items:center;gap:6px">'
        f'<b style="font-family:{MONO};color:oklch(0.5 0.14 25)">!</b> still flagged</span>'
        f'<span style="display:flex;align-items:center;gap:6px">'
        f'<b style="font-family:{MONO};color:oklch(0.45 0.13 150)">✓</b> a human ruled on it</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for student_id in sorted(by_student):
        cells = []
        row_total = 0
        for key in keys:
            item = by_student[student_id].get(key)
            if item is None:
                cells.append(f'<td style="color:{INK_4}">·</td>')
                continue
            row_total += item.grade or 0
            mark = "✓" if item.decided else ("!" if item.flags(limit) else "")
            text = "—" if item.grade is None else str(item.grade)
            flags = ", ".join(FLAG_TEXT[f] for f in item.flags(limit)) or "no flags"
            # A link rather than a button: 243 Streamlit buttons cost more to
            # render than the whole rest of this view, and a query param is the
            # only jump the browser can make without a round-trip per cell.
            cells.append(
                f'<td><a class="cell" href="?goto={html.escape(student_id)}::{html.escape(key)}"'
                f' target="_self" title="{html.escape(student_id)} {html.escape(key)} — {html.escape(flags)}"'
                f' style="background:{grade_colour(item.grade, .7)};color:{grade_ink(item.grade)}">'
                f'{text}<sub>{mark}</sub></a></td>'
            )
        rows.append(
            f'<tr><td class="sid" dir="auto">{html.escape(student_id)}</td>'
            f'{"".join(cells)}<td class="tot" style="color:{INK_2}">{row_total}</td></tr>'
        )

    header = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    st.markdown(
        f'<div class="card"><div class="wrap"><table class="grid"><thead><tr>'
        f'<th class="sid">student</th>{header}<th>total</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h2 class="view">Distribution per item</h2>'
        '<p class="view">A part averaging near 0 or near 100 usually means the marking '
        'scheme was misread, not that the cohort was uniform.</p>',
        unsafe_allow_html=True,
    )
    cards = []
    for key in keys:
        values = [i.grade for i in session.items if i.key == key and i.grade is not None]
        withheld = sum(1 for i in session.items if i.key == key and i.grade is None)
        buckets = [0] * 10
        for v in values:
            buckets[min(9, v // 10)] += 1
        top = max(buckets) or 1
        bars = "".join(
            f'<i title="{b} students scored {i * 10}–{i * 10 + 9}" '
            f'style="height:{max(2, round(b / top * 70))}px;'
            f'background:{grade_colour(i * 10 + 5, .85)}"></i>'
            for i, b in enumerate(buckets)
        )
        mean = f"{sum(values) / len(values):.1f}" if values else "—"
        cards.append(
            f'<div class="card" style="padding:14px">'
            f'<div style="display:flex;align-items:baseline;justify-content:space-between;'
            f'margin-bottom:10px"><b style="font-size:13.5px">{html.escape(key)}</b>'
            f'<span style="font-family:{MONO};font-size:11.5px;color:{INK_3}">'
            f'mean {mean} · n={len(values)}</span></div>'
            f'<div class="bars">{bars}</div>'
            f'<div class="axis"><span>0</span><span>50</span><span>100</span></div>'
            + (f'<div style="font-size:11px;color:{INK_3};margin-top:8px">'
               f'{withheld} withheld / ungraded</div>' if withheld else "")
            + "</div>"
        )
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));'
        'gap:12px">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )

    tags: Dict[str, Dict[str, int]] = {}
    for item in session.items:
        if item.tag:
            tags.setdefault(item.key, {}).setdefault(item.tag, 0)
            tags[item.key][item.tag] += 1
    if tags:
        st.markdown(
            '<h2 class="view">Labels recorded</h2>'
            '<p class="view">Descriptive only — a label never affects a mark.</p>',
            unsafe_allow_html=True,
        )
        for key, counts in tags.items():
            st.write(f"**{key}** — " + ", ".join(
                f"`{name}` {n}" for name, n in sorted(counts.items(), key=lambda t: -t[1])
            ))


# --- Fairness -----------------------------------------------------------------

def view_fairness(session: ReviewSession) -> None:
    st.markdown(
        '<h1 class="view">Same answer, different mark</h1>'
        '<p class="view">Pairs of submissions whose evidence quotes are near-identical '
        'but whose marks are not. These are grading errors by definition — no ground '
        'truth and no trust in the model is required to read this view.</p>',
        unsafe_allow_html=True,
    )
    with_evidence = sum(1 for i in session.items if i.evidence)
    if not with_evidence:
        st.info(
            "This results file carries no evidence quotes, so consistency cannot be "
            "checked. Runs made after `assurance.py` landed record a verbatim quote "
            "per part; re-grading with `--samples` or any current build will populate it."
        )
        return

    c1, c2, c3 = st.columns([3, 3, 1.4])
    similarity = c1.slider("How similar the answers must be", 0.70, 1.0, 0.90, 0.01)
    gap = c2.slider("Mark gap worth reporting", 5, 60, 15, 5)
    pairs = session.inconsistencies(similarity=similarity, gap=gap)
    c3.markdown(
        f'<div style="text-align:end;padding-top:8px">'
        f'<div style="font-family:{MONO};font-size:24px;font-weight:600;line-height:1">'
        f'{len(pairs)}</div><div style="font-size:10.5px;text-transform:uppercase;'
        f'letter-spacing:.07em;color:{INK_3};font-weight:600">pairs found</div></div>',
        unsafe_allow_html=True,
    )

    if not pairs:
        st.markdown(
            f'<div style="padding:26px;text-align:center;border:1px dashed #d3d0c8;'
            f'border-radius:11px;color:{INK_2};font-size:13px">No pairs above '
            f'{similarity:.0%} similarity differ by {gap} marks or more.</div>',
            unsafe_allow_html=True,
        )
        return

    # Two-up so the near-identical quotes are physically adjacent. That
    # adjacency is the whole argument the view makes.
    cards = []
    for a, b, ratio in pairs[:40]:
        pair_gap = abs((a.grade or 0) - (b.grade or 0))
        sides = []
        for i, side in enumerate((a, b)):
            sides.append(
                f'<div style="padding:13px 15px;'
                f'{"border-inline-end:1px solid #f0eee8" if i == 0 else ""}">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'gap:8px;margin-bottom:8px">'
                f'<a dir="auto" href="?goto={html.escape(side.student_id)}::{html.escape(side.key)}"'
                f' target="_self" style="font-family:{MONO};font-size:12.5px;font-weight:600">'
                f'{html.escape(side.student_id)}</a>'
                f'<span style="font-family:{MONO};font-size:14px;font-weight:700;'
                f'padding:2px 9px;border-radius:6px;background:{grade_colour(side.grade, .6)};'
                f'color:{grade_ink(side.grade)}">{"—" if side.grade is None else side.grade}</span>'
                f'</div>'
                f'<div dir="auto" style="font-family:{MONO};font-size:12px;line-height:1.7;'
                f'color:#3a382f">{html.escape(side.evidence)}</div></div>'
            )
        cards.append(
            f'<div class="card"><div class="pairhead">'
            f'<b style="font-size:13px">{html.escape(a.label)}</b>'
            f'<span style="{chip_style(25 if pair_gap >= 30 else 45)}">gap {pair_gap}</span>'
            f'<span style="font-family:{MONO};font-size:11.5px;color:{INK_3}">'
            f'{ratio:.0%} similar</span></div>'
            f'<div class="pair">{"".join(sides)}</div></div>'
        )
    st.markdown(
        '<div style="display:flex;flex-direction:column;gap:10px">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


# --- Ledger and sign-off (unchanged behaviour) --------------------------------

def view_ledger(session: ReviewSession) -> None:
    records = [r for r in session.ledger() if "student_id" in r]
    st.markdown(
        '<h1 class="view">Decision ledger</h1>'
        '<p class="view">Every human ruling, appended and never overwritten. This is '
        'the ground truth this project has never had: the model’s grade and a human’s, '
        'side by side, accumulating over a term at no cost.</p>',
        unsafe_allow_html=True,
    )
    if not records:
        st.info("No decisions recorded yet.")
        return

    changed = [r for r in records if r.get("was") != r.get("now") and r.get("reason") != "undo"]
    moves = [abs((r["now"] or 0) - (r["was"] or 0))
             for r in changed if r.get("now") is not None and r.get("was") is not None]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisions", len(records))
    c2.metric("Changed", len(changed))
    c3.metric("Mean change", f"{sum(moves) / len(moves):.1f}" if moves else "—")
    c4.metric("Withheld by hand", sum(1 for r in records if r.get("now") is None))

    blind = [
        r for r in records
        if r.get("blind") and r.get("blind_grade") is not None
        and r.get("model_grade") is not None
    ]
    st.markdown('<h2 class="view">Measured accuracy</h2>', unsafe_allow_html=True)
    if not blind:
        st.info(
            "No blind marks recorded yet. Turn on **blind mode** in the top bar and "
            "the ordinary work of reviewing produces the one number this project "
            "does not have: how far the model's grades sit from a human's, measured "
            "on marks given before the model's were visible.\n\n"
            "Thirty items is enough to see an effect size."
        )
    else:
        errors = [r["model_grade"] - r["blind_grade"] for r in blind]
        absolute = [abs(e) for e in errors]
        mae = sum(absolute) / len(absolute)
        bias = sum(errors) / len(errors)
        within10 = sum(1 for e in absolute if e <= 10) / len(absolute)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Blind marks", len(blind))
        c2.metric("Mean absolute error", f"{mae:.1f}",
                  help="Average distance between the model's grade and a human's, "
                       "in marks out of 100.")
        c3.metric("Bias", f"{bias:+.1f}",
                  help="Positive means the model marks more generously than you do.")
        c4.metric("Within 10 marks", f"{within10:.0%}")
        st.caption(
            "Measured on the items a human happened to review, which are the ones "
            "the queue flagged as doubtful — so this is the model's accuracy on its "
            "*hardest* cases, not on the cohort. It is a floor, not an average, and "
            "it should be reported that way."
        )
        if len(blind) < 30:
            st.caption(f"{30 - len(blind)} more blind marks to reach a usable sample.")

    high = [r for r in records if r.get("confidence") == "high"]
    if high:
        overturned = sum(1 for r in high if r.get("was") != r.get("now"))
        st.metric(
            "High-confidence grades a human changed",
            f"{overturned}/{len(high)} ({overturned / len(high):.0%})",
            help="Confidence calibration. If the system's high-confidence grades are "
                 "routinely overturned, the confidence signal is broken and should be "
                 "said so rather than displayed.",
        )

    st.dataframe(
        [
            {
                "student": r.get("student_id"),
                "item": r.get("label") or r.get("item"),
                "was": r.get("was"),
                "now": r.get("now"),
                "reason": r.get("reason"),
                "note": r.get("note"),
                "by": r.get("by"),
                "at": r.get("at"),
            }
            for r in reversed(records)
        ],
        width="stretch", hide_index=True,
    )
    st.download_button(
        "Download ledger (JSONL)",
        data=session.ledger_path.read_text(encoding="utf-8") if session.ledger_path.exists() else "",
        file_name=session.ledger_path.name,
        mime="application/x-ndjson",
    )


def view_signoff(session: ReviewSession) -> None:
    st.markdown(
        '<h1 class="view">Sign-off and export</h1>'
        '<p class="view">The system produces a proposal; a human is the author of '
        'record. Nothing here is final until someone accepts the batch, and the export '
        'says whether they did.</p>',
        unsafe_allow_html=True,
    )

    blocking = session.blocking_items()
    decided = sum(1 for i in session.items if i.decided)
    c1, c2, c3 = st.columns(3)
    c1.metric("Items", len(session.items))
    c2.metric("Decided", decided)
    c3.metric("Unresolved", len(blocking))

    if blocking:
        st.error(
            f"{len(blocking)} item(s) have no publishable grade — withheld, or never "
            "graded. Publishing now records a blank or a zero against work nobody "
            "has looked at."
        )
        with st.expander("Show them"):
            for item in blocking[:60]:
                why = VETO_TEXT.get(item.veto, item.veto) if item.veto else "no grade produced"
                st.write(f"- **{item.student_id}** {item.label} — {why}")

    signed = session.signed_off
    if signed:
        st.success(
            f"Signed off by **{signed['by']}** at {signed['at']} — "
            f"{signed['reviewed']}/{signed['items']} items reviewed, "
            f"{signed['unresolved']} unresolved."
            + (f"\n\n> {signed['note']}" if signed.get("note") else "")
        )

    note = st.text_input("Sign-off note", placeholder="e.g. part (c) spot-checked against pages")
    if st.button("Sign off on this batch", type="primary"):
        record = session.sign_off(note)
        st.success(f"Signed off — {record['reviewed']}/{record['items']} items.")
        st.rerun()

    st.divider()
    default_name = session.path.with_suffix("").name + ".csv"
    name = st.text_input("Export filename", value=default_name)
    if st.button("Write CSV", disabled=not session.signed_off,
                 help=None if session.signed_off else "Sign off first."):
        path = session.export_csv(name)
        st.success(f"Wrote {path}")
    if not session.signed_off:
        st.caption("Export is gated on sign-off deliberately — see the note above.")
    if Path(name).exists():
        st.download_button(
            "Download the CSV", data=Path(name).read_bytes(),
            file_name=name, mime="text/csv",
        )
        st.caption(
            "This file contains student IDs and marks. It is gitignored and must stay "
            "that way."
        )


# --- App ----------------------------------------------------------------------

def handle_goto() -> None:
    """
    `?goto=<student>::<part>` — the link out of Cohort and Fairness into Triage.

    That link was the thing missing between "I can see the problem" and "I can
    fix it": both views could show an inconsistency and neither could open it.
    The search box is the vehicle because a fully-decided student is not in the
    priority queue and so has no cursor position to jump to.
    """
    raw = st.query_params.get("goto")
    if not raw:
        return
    student_id, _, key = str(raw).partition("::")
    st.session_state.view = "Triage"
    st.session_state.student_search = student_id
    st.session_state.active_key = key or None
    st.session_state.pop("_draft_for", None)
    st.query_params.clear()


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    handle_goto()

    files = sorted(glob.glob(os.path.join(RESULTS_FOLDER, "grades_*.json")))
    if not files:
        st.error("No results files found (grades_*.json) in this directory.")
        return

    reviewer = st.session_state.get("reviewer") or os.environ.get("USER", "")
    probe = get_session(st.session_state.get("results_file") or files[0], reviewer)
    path = render_topbar(probe, files)
    session = get_session(path, reviewer)

    view = st.session_state.get("view", "Triage")
    pairs = 0
    if any(i.evidence for i in session.items):
        try:
            pairs = len(session.inconsistencies(similarity=0.90, gap=15))
        except Exception:
            pairs = 0
    render_rail(view, session.triage()["pending"], pairs)

    with st.sidebar:
        st.markdown("### Setup")
        st.caption(
            "Everything here changes how the queue is built, not what is in front "
            "of you. It is deliberately off the hot path."
        )
        st.text_input("Reviewer", key="reviewer",
                      value=reviewer, help="Recorded against every decision.")
        if session.file_changed_on_disk():
            st.warning(
                "This file has changed on disk since it was opened — a grading run is "
                "probably still writing to it. Reload before deciding anything, or "
                "your next save will overwrite its new rows."
            )
        if st.button("Reload from disk", width="stretch"):
            session.reload()
            st.rerun()

        st.divider()
        st.markdown("**Marks per item**")
        st.caption("What each part is worth. This sets the queue order — a contested "
                   "grade on a 4-mark part outranks one on a 1-mark part.")
        weights = {}
        cols = st.columns(min(len(session.item_keys), 4) or 1)
        for i, key in enumerate(session.item_keys):
            weights[key] = cols[i % len(cols)].number_input(
                key, min_value=0.0, value=float(session.state.get("weights", {}).get(key, 1.0)),
                step=0.5, key=f"w_{key}",
            )
        if weights != {k: float(v) for k, v in session.state.get("weights", {}).items()}:
            session.set_weights(weights)

        st.divider()
        folders = candidate_pdf_folders()
        chosen = st.multiselect(
            "Submission folders", folders, default=folders,
            help="Where to look for <student_id>.pdf. Per-assignment folders are "
                 "picked up automatically.",
        )
        st.toggle("Show the shortcut strip", key="show_keys_strip", value=True)

        st.divider()
        counts = session.flag_counts()
        if counts:
            st.markdown("**Flags across the file**")
            for flag, n in sorted(counts.items(), key=lambda t: -t[1]):
                st.write(f"- {FLAG_TEXT[flag]}: **{n}**")
        posture = session.data.get("posture")
        if isinstance(posture, dict):
            with st.expander("Grading posture used"):
                st.json(posture)
        if session.data.get("usage"):
            with st.expander("What this run cost"):
                st.json(session.data["usage"])

    if view == "Triage":
        view_triage(session, chosen)
    else:
        # Triage owns the whole viewport; every other view is a page with
        # margins and ordinary Streamlit spacing.
        with st.container(key="viewbody"):
            if view == "Cohort":
                view_cohort(session)
            elif view == "Fairness":
                view_fairness(session)
            elif view == "Ledger":
                view_ledger(session)
            else:
                view_signoff(session)

    if st.session_state.get("keys_open"):
        render_keys_panel()

    toast = st.session_state.pop("toast", "")
    if toast:
        st.markdown(f'<div class="toast">{html.escape(toast)}</div>', unsafe_allow_html=True)

    # Last, so the buttons it clicks are all in the DOM by the time it binds.
    components.html(
        KEYS_JS.substitute(
            keys_open="true" if st.session_state.get("keys_open") else "false",
            cited_token=json.dumps(st.session_state.get("cited_token", "")),
        ),
        height=0,
    )


if __name__ == "__main__":
    main()
