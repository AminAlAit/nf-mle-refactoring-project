# Project conventions

These apply to **Amin's files only**: anything named `*_amin.py`, plus
`kc/__init__.py`, `app/__init__.py`, `tests/conftest.py`, and the markdown and
notebook files at the repo root.

They do **not** apply to:

- `attempts/**` (teammates' work, kept exactly as they wrote it)
- `bonus_solution/**` (course reference material)
- `King-County.ipynb`, `project-for-today.md` (the original template)

Never reformat, rename, or tidy a file in those paths. The whole point of keeping
them is to compare what each person actually wrote.

## No em-dashes

Not anywhere, strictly. Neither the em-dash character itself nor ` -- ` standing
in for one.

Use a comma, a colon, a full stop, or brackets instead:

```
wrong:  Derived from price -- it would leak the answer.
right:  Derived from price. It would leak the answer.
right:  Derived from price, so it would leak the answer.
```

`--flag` on a command line is fine. So is `---` as a markdown rule or table
separator. The rule is about dashes used as punctuation in prose.

## Docstrings start on their own line

The text begins on the line *after* the opening quotes, never on the same line.
This holds for one-line docstrings too, and for module and class docstrings.

```python
# wrong
def load_raw(path):
    """Read the CSV, no cleaning."""

# right
def load_raw(path):
    """
    Read the CSV, no cleaning.
    """
```

## Keep docstrings short

One line where one line does the job, and plenty of functions need none at all.
A docstring explains *why* something is odd, not *what* the code plainly says.
Do not write a paragraph where a clause works.

```python
# too much
def drop_invalid_rows(df, max_bedrooms=30):
    """
    Remove records that cannot describe a real house.

    Notebook cells 26-28 remove one row listing 33 bedrooms, 2 bathrooms and
    1620 sqft. The notebook removes it by its position in the file, which breaks
    if the frame was filtered, sorted or re-indexed first, so we remove it by the
    property that makes it wrong instead.
    """

# right
def drop_invalid_rows(df, max_bedrooms=30):
    """
    Drop the 33-bedroom house. Cells 26-28.
    """
```

## File naming

Our modules end in `_amin` so a teammate's take on the same part can sit beside
them. Three files keep plain names because the tooling demands it:

| File | Why |
|---|---|
| `kc/__init__.py`, `app/__init__.py` | Python needs this exact name for a package |
| `tests/conftest.py` | pytest only discovers fixtures from this exact name |

## Commit messages are one line

A subject line and nothing else. No body, no bullets, no trailers.

```
right:  Add feature engineering and vectorise the water distance loop
wrong:  the same thing followed by six paragraphs explaining it
```

Reasoning belongs in the code comments or in `REFACTORING.md`, where it stays
readable. Amin writes the commits himself, so hand him exactly one line.

## Before saying something works

Run it. `uv run pytest`, `uv run python -m kc.train_amin`, or an actual request
against the running API. A command exiting 0 through a pipe is not evidence,
and neither is a green build log on its own.
