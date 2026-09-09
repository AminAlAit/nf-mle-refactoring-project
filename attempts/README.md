# Attempts

Three of us worked on this project and we deliberately overlapped, because
splitting it four ways would have meant each person only ever learning a quarter of
it. So the same parts were solved more than once, on purpose, and this folder is
where the other solutions live side by side.

## The convention

| Location | Whose |
|---|---|
| repo root (`kc/`, `app/`, `tests/`) | Amin's, the complete run through every part |
| `attempts/<name>/` | a teammate's work, exactly as they wrote it |

The root version is complete on its own: cleaning, feature engineering, the
pipeline, modeling, the API, Docker and tests. Nothing here is needed for it to run.

## Adding a teammate's work

**If they pushed a branch:**

```bash
git fetch origin
git checkout -b merge/<name> feat/full-refactor
git read-tree --prefix=attempts/<name>/ -u origin/<their-branch>
git commit -m "Add <name>'s attempt"
```

`read-tree --prefix` drops their whole tree into `attempts/<name>/` instead of
merging it onto ours, so two solutions to the same file sit next to each other
rather than fighting over the same path. No conflicts to resolve, and nothing of
theirs is rewritten to match our naming.

**If they sent files:** copy them into `attempts/<name>/` as they are. Do not
reformat, rename, or "fix" them. The point is to compare what each person actually
wrote.

## Adding a note

Each attempt folder should get a short `NOTES.md` saying which parts that person
covered and anything they did differently that is worth stealing. Comparing two
solutions to the same problem is most of the value here; a diff with no commentary
is much harder to learn from.
