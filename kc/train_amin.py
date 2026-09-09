"""
Train, tune, evaluate, save.

    python -m kc.train_amin            full run with the grid search
    python -m kc.train_amin --quick    skip the grid search
"""

import argparse
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV

from kc.cleaning_amin import clean
from kc.config_amin import CV_FOLDS, ELASTICNET_PARAM_GRID, METRICS_PATH, MODEL_PATH
from kc.data_amin import load_raw
from kc.features_amin import add_sqft_price
from kc.modeling_amin import (
    error_table,
    evaluate,
    save_metrics,
    save_model,
    split_dataset,
)
from kc.pipeline_amin import build_baseline_pipeline, build_model_pipeline

# The notebook's two baselines, cells 88 and 95.
BASELINE_VARIABLES = [
    ["grade"],
    ["grade", "last_known_change"],
]


def prepare_dataset() -> pd.DataFrame:
    """
    Raw CSV to the frame that gets split.

    Row-dropping has to happen before the split, and sqft_price is for analysis
    only, so the pipeline excludes it.
    """
    return add_sqft_price(clean(load_raw()))


def run(quick: bool = False) -> list[dict]:
    """
    Train everything, return the scores.
    """
    started = time.perf_counter()

    print("Loading and cleaning...")
    dataset = prepare_dataset()
    raw_rows = len(load_raw())
    print(f"  {raw_rows} rows in, {len(dataset)} after cleaning "
          f"({raw_rows - len(dataset)} dropped)")

    X_train, X_test, y_train, y_test = split_dataset(dataset)
    print(f"  train {X_train.shape[0]} rows / test {X_test.shape[0]} rows")

    metrics: list[dict] = []

    print("\nBaselines...")
    for variables in BASELINE_VARIABLES:
        baseline = build_baseline_pipeline(variables).fit(X_train, y_train)
        score = evaluate(
            baseline, X_test, y_test, label=f"linear({'+'.join(variables)})"
        )
        metrics.append(score)
        print(f"  {score['label']:<34} adj R^2 {score['adjusted_r2']:.3f}")

    if quick:
        print("\nElasticNet (quick, no grid search)...")
        model = build_model_pipeline().fit(X_train, y_train)
        best_params = {"note": "quick run, defaults used"}
    else:
        combinations = len(ELASTICNET_PARAM_GRID["model__regressor__alpha"]) * len(
            ELASTICNET_PARAM_GRID["model__regressor__l1_ratio"]
        )
        print(f"\nElasticNet grid search "
              f"({combinations} combinations x {CV_FOLDS} folds)...")
        search = GridSearchCV(
            build_model_pipeline(),
            ELASTICNET_PARAM_GRID,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        print(f"  best: {best_params}")
        print(f"  best cross-validated R^2: {search.best_score_:.3f}")

    score = evaluate(model, X_test, y_test, label="elasticnet(poly2)")
    score["best_params"] = {str(k): v for k, v in best_params.items()}
    metrics.append(score)
    print(f"  {score['label']:<34} adj R^2 {score['adjusted_r2']:.3f} "
          f"| R^2 {score['r2']:.3f} | RMSE ${score['rmse']:,.0f}")

    print("\nLargest over-prediction on the test set:")
    errors = error_table(model, X_test, y_test)
    worst = errors.loc[errors["price_difference_percent"].idxmax()]
    print(f"  id {int(worst['id'])}: sold for ${worst['price']:,.0f}, "
          f"predicted ${worst['price_prediction']:,.0f} "
          f"({worst['price_difference_percent']:.0f}% over)")

    save_model(model, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)
    print(f"\nSaved model/{MODEL_PATH.name} and model/{METRICS_PATH.name}")
    print(f"Total time: {time.perf_counter() - started:.1f}s")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Skip the grid search."
    )
    run(quick=parser.parse_args().quick)


if __name__ == "__main__":
    main()
