FROM ghcr.io/astral-sh/uv:0.12.7 AS uv
FROM python:3.13-slim-bookworm

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY --from=uv /uv /uvx /bin/

# Dependencies before source, so editing a Python file does not invalidate the
# cached dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

# `kc` is needed at runtime, not only for training: the saved pipeline
# deserialises into the transformer classes defined there.
COPY kc ./kc

# `model/` is a build artifact and is gitignored, so it cannot simply be copied in
# -- a fresh clone has no model file. Training it here instead takes about eight
# seconds, is reproducible because the split and the grid are seeded, and means
# `docker compose up --build` yields a container whose /predict endpoint works
# with no prior local setup. The dataset is removed in the same layer so it does
# not add 2.4 MB to the image.
COPY data ./data
RUN uv run --no-sync python -m kc.train_amin && rm -rf data

COPY app ./app

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main_amin:app", "--host", "0.0.0.0", "--port", "8000"]
