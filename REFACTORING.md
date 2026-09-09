# Notes on my refactor

What I moved out of [King-County.ipynb](King-County.ipynb) and why.

## Running it

```bash
uv sync
uv run pytest                              # 54 tests
uv run python -m kc.train_amin             # saves model/model.bin
uv run uvicorn app.main_amin:app --reload  # http://localhost:8000/docs
```

Or in Docker, which trains the model while building the image so `/predict` works
straight away:

```bash
cp .env.example .env
docker compose up --build --wait
```

## Layout

```
kc/
  config_amin.py          constants
  data_amin.py            loading the CSV            (cell 7)
  cleaning_amin.py        data cleaning              (cells 26-43)
  features_amin.py        the derived columns        (cells 56-70)
  transformers_amin.py    sklearn wrappers
  pipeline_amin.py        the pipelines              (cells 100-117)
  modeling_amin.py        split, score, save, load   (cells 78-131)
  train_amin.py           python -m kc.train_amin
  predict_amin.py         scoring a new house
app/                      FastAPI, CRUD plus /predict
tests/                    54 tests
King-County-refactored.ipynb
```

I left the original notebook and `bonus_solution/` alone so the before and after
can be compared.

## Three things I decided

**Cleaning is split in two.** `clean_rowwise` only changes values inside rows.
`drop_invalid_rows` removes rows. They are separate because a step that drops rows
cannot go inside a sklearn pipeline: X and y end up different lengths and nothing
warns you. So row-dropping runs once before the split, and everything else lives
inside the model, which means it also runs on API requests without me writing it
twice.

**Nothing modifies its input.** The notebook uses `inplace=True` in cells 28 and
43, so you cannot re-run it from the top. Every function here returns a new frame,
and `clean(clean(df)) == clean(df)` is a test.

**The saved model carries its own preprocessing.** `model/model.bin` is the whole
pipeline. Give it a frame with the raw CSV columns and it gives back a price. That
is why `/predict` can take plain house attributes instead of me reimplementing
nineteen features in the API.

## Two bugs I found in the notebook

**A data leak.** Cell 68 builds the waterfront list from the whole dataset and
cell 69 uses it, both before the split in cell 82. So test rows help build a
training feature. My `WaterDistance` learns that list in `fit`, from training rows
only. It also has to work that way for the API, because one house on its own has
no waterfront neighbours to measure against.

**A very slow loop.** Cell 69 loops over 146 waterfront houses inside a loop over
all 21,596 rows, which takes about 70 seconds. The same arithmetic as one NumPy
broadcast takes 0.09s. I kept the old loop in `water_distance_naive` so a test can
run both on 200 real rows and check they give identical results.

## Results

From `python -m kc.train_amin`:

| Model | Features | Adjusted R2 | RMSE |
|---|---|---|---|
| Linear, `grade` | 1 | 0.432 | $274,288 |
| Linear, `grade` + `last_known_change` | 2 | 0.480 | $262,320 |
| ElasticNet, degree 2 | 209 | 0.840 | $143,443 |

The first two match the 43% and 48% the notebook reports in cells 91 and 98. Grid
search picked `alpha=0.01`, `l1_ratio=0.2`.

I kept RMSE next to R2 on purpose. The model explains 84% of the variance and is
still usually off by about $143,000, and the second number is the one that matters
if you are actually bidding on a house.

## Other things I noticed

- The notebook disagrees with itself about the target. The code fits
  `y = kc_data.price` in cell 81, but the text says "price per square foot" in
  cells 91 and 98. I went with total price, matching the code.
- `sqft_basement` has 454 `?` entries, not one. Recomputing it from
  `sqft_living - sqft_above` handles them all, but the text makes the problem
  sound smaller than it is.
- Cell 72 says houses average almost 19 km from a waterfront house. Running the
  notebook's own code gives 5.5 km. The outputs were cleared so I cannot check it
  against a real run, but my version and the original loop agree exactly.
- `zipcode` is used as a number, so the model can think 98104 sits between 98103
  and 98105. It is a label. One-hot encoding it would probably help.
- `sqft_above + sqft_basement` is exactly `sqft_living`, so three fully dependent
  columns go into the polynomial expansion.
- The centre-of-wealth formula measures the offset from 47.62774 but corrects
  longitude with cos(47.6219). The gap is tiny so I kept the notebook's values
  rather than quietly changing the numbers.
