# The Refactor

What moved out of [King-County.ipynb](King-County.ipynb), where it went, and why.

## Quick start

```bash
uv sync                                    # install
uv run pytest                              # 54 tests, ~9s
uv run python -m kc.train_amin             # trains and saves model/model.bin, ~8s
uv run uvicorn app.main_amin:app --reload  # API on http://localhost:8000/docs
```

With Docker (trains the model during the image build, so `/predict` works immediately):

```bash
cp .env.example .env
docker compose up --build --wait
```

## Layout

Every file of ours ends in `_amin` so a teammate's version of the same part can sit
next to it. Three files keep their plain names because the tooling requires it:
`kc/__init__.py`, `app/__init__.py` (Python) and `tests/conftest.py` (pytest).

```
kc/                       the refactored logic
  config_amin.py          every constant the notebook hardcoded inline
  data_amin.py            reading the CSV                      (cell 7)
  cleaning_amin.py        fixing the raw data                  (cells 26-43)
  features_amin.py        the three derived columns            (cells 56-70)
  transformers_amin.py    scikit-learn wrappers around both
  pipeline_amin.py        the assembled pipelines              (cells 100-117)
  modeling_amin.py        split, score, save, load             (cells 78-131)
  train_amin.py           `python -m kc.train_amin`
  predict_amin.py         loading a model and pricing a house
app/                      FastAPI: CRUD over houses + /predict
tests/                    54 tests
King-County-refactored.ipynb   the analysis, using kc/
```

The original notebook and `bonus_solution/` are untouched, so the before and after
can be compared directly.

## The three decisions that shaped it

### 1. Cleaning is split by whether it drops rows

`clean_rowwise` only changes values inside existing rows. `drop_invalid_rows`
removes them. They are separate because only the first is safe inside a
scikit-learn pipeline. Dropping a row mid-pipeline leaves `X` and `y` misaligned
with no error, and the model trains on mismatched pairs.

So row-dropping runs once, before the split. Everything else lives inside the model
and therefore also runs on incoming API requests, with no second implementation to
keep in sync.

### 2. Every function returns a new frame

The original uses `inplace=True` (cells 28, 43) and drops a row by its position in
the file (`kc_data.drop(15856)`). Both make a cell destructive: re-running the
notebook from the top after an edit deletes a *different* house, silently.

Here nothing mutates its input, the 33-bedroom record is removed by the condition
that makes it wrong, and `clean(clean(df)) == clean(df)` is a test.

### 3. The saved model carries its own preprocessing

`model/model.bin` is the entire pipeline: cleaning, feature engineering, column
selection, polynomial expansion, scaling, and the estimator. Hand it a frame with
the raw CSV columns and it returns a price.

That is what allows `/predict` to accept plain house attributes. The alternative,
re-implementing nineteen features in the serving code, is the standard way for
training and serving to drift apart.

## Two bugs found in the original

**A data leak.** Cell 68 builds the waterfront reference list from the whole
dataset, then cell 69 derives `water_distance` from it, before the train/test
split in cell 82. Test rows therefore help shape a training feature.
`transformers_amin.WaterDistance` learns the reference set in `fit`, so it sees training
rows only. It also has to work this way for a different reason: a single house
arriving at the API has no waterfront neighbours of its own to measure against.

**A 400x slower loop than necessary.** Cell 69 nests a Python loop over the 146
waterfront houses inside a loop over all 21,596 houses. `features_amin.water_distance`
does the same arithmetic as one NumPy broadcast: 0.09s instead of roughly 70s.
`tests/test_features_amin.py` runs the original loop against the new one on 200 real
rows and asserts the results are *identical*, not merely close. The old
implementation is kept in `features_amin.water_distance_naive` purely so that test can
exist.

## Results

Reproduced from `python -m kc.train_amin`:

| Model | Features | Adjusted R² | RMSE |
|---|---|---|---|
| Linear, `grade` only | 1 | 0.432 | $274,288 |
| Linear, `grade` + `last_known_change` | 2 | 0.480 | $262,320 |
| ElasticNet, degree-2 polynomial | 209 | **0.840** | $143,443 |

The first two match the notebook's stated 43% and 48% (cells 91, 98). Best
parameters found by grid search: `alpha=0.01`, `l1_ratio=0.2`.

The RMSE column is there on purpose. A model that explains 84% of the variance is
still typically wrong by about $143,000, and only one of those two facts is useful
when deciding what to bid.

## Things in the original worth knowing about

- **The notebook contradicts itself about the target.** The code fits
  `y = kc_data.price` (cell 81); the surrounding text says "variance in price per
  square foot" three times (cells 91, 98). This refactor fits total price, matching
  the code. Worth settling deliberately rather than by accident.
- **`sqft_basement` has 454 `?` entries**, not one. The notebook's fix, recomputing
  from `sqft_living - sqft_above`, handles all of them, but the prose
  undersells the scale of the problem.
- **The notebook's "almost 19 km" average water distance** (cell 72) does not match
  its own code, which produces 5.5 km. The stored outputs were cleared, so the
  figure cannot be checked against a real run; the loop in cell 69 and the
  vectorised version here agree exactly, so 5.5 km is what that code computes.
- **`zipcode` is treated as a number**, so the model can infer that 98104 sits
  "between" 98103 and 98105. It is a label, and encoding it as one would likely help.
- **`sqft_above + sqft_basement == sqft_living`** exactly, by construction. Three
  perfectly dependent columns then go through a polynomial expansion.
- **The centre-of-wealth formula uses two different latitudes**: it measures the
  offset from 47.62774 but corrects longitude with cos(47.6219). The difference is
  negligible (7e-5 relative), and `kc/config_amin.py` keeps the notebook's values so the
  numbers stay comparable rather than silently "fixed".
