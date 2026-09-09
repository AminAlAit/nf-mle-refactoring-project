# model persistence: saving and loading a fitted pipeline

from __future__ import annotations

from pathlib import Path

import skops.io as sio
from sklearn.pipeline import Pipeline


def save_model(model: Pipeline, path: Path) -> None:
    """
    Saves a fitted pipeline to disk, so it can be reloaded later with `load_model`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f_out:
        sio.dump(model, f_out)


def load_model(path: Path) -> Pipeline:
    """
    Loads a fitted pipeline from disk, which was previously saved with `save_model`.
    """
    untrusted_types = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=untrusted_types)