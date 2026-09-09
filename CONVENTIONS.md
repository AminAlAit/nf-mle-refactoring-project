# Conventions

How I write code in this repo. Only applies to my branch. The other two branches
are my teammates' work and nobody should reformat them.

## No em-dashes

Use a comma, a colon, a full stop or brackets instead. `--flag` on a command line
is fine, and so is `---` in markdown.

## Docstrings

Text starts on the line after the quotes:

```python
def load_raw(path):
    """
    Read the CSV, no cleaning.
    """
```

Keep them short. One line usually, and plenty of functions need none. A docstring
is for saying why something is odd, not repeating what the code says.

## Commits

One line, no body.

## Before claiming something works

Run it. `uv run pytest`, `uv run python -m kc.train`, or an actual request
against the API. A command exiting 0 is not the same as the thing working.
