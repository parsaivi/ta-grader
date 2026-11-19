# Contributing

Bug reports, locale packs and fixes are welcome.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check src tests
```

Tests run offline with no API key. Please add one for any behaviour you
change.

**Never commit student data.** Use the synthetic demo (`ta-grader demo`) or
invent submissions for tests and bug reports. If a bug needs a real
submission to reproduce, describe it instead of attaching it.

Before changing how grades are produced or recorded, read
[docs/design.md](docs/design.md). Several rules there look like they could be
simplified and can't: each one is a bug that reached real students once.
In particular, a pull request must not:

- record `0` where there is no grade (`null` and `0` mean different things),
- read a grade out of free text instead of the explicit `GRADE:` line,
- let a posture or tag scheme file add free text to the prompt,
- let anything export grades without a sign-off.

A new language is a TOML file: see [docs/adding-a-language.md](docs/adding-a-language.md).
