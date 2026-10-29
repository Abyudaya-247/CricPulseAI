import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "chase_training_data.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "chronological_evaluation.csv"

# Train on seasons before 2025; test on 2025 and later.
TEST_START_YEAR = 2025

FEATURES = [
    "target_runs",
    "current_score",
    "runs_required",
    "wickets_remaining",
    "balls_remaining",
    "current_run_rate",
    "required_run_rate",
]

TARGET = "chase_win"
GROUP = "match_id"


def evaluate(actual, probabilities, predictions):
    return {
        "accuracy": accuracy_score(actual, predictions),
        "roc_auc": roc_auc_score(actual, probabilities),
        "log_loss": log_loss(
            actual, probabilities, labels=[0, 1]
        ),
        "brier_score": brier_score_loss(actual, probabilities),
    }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run python src/build_training_data.py first."
        )

    df = pd.read_csv(DATA_FILE, dtype={"season": str})

    required = FEATURES + [TARGET, GROUP, "season"]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Extract the first four-digit year from season labels.
    # Examples: 2007/08 -> 2007, 2020/21 -> 2020.
    df["season_year"] = pd.to_numeric(
        df["season"].str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )

    df = df.dropna(
        subset=["season_year", TARGET, GROUP]
    ).copy()

    df["season_year"] = df["season_year"].astype(int)

    # The split is based on season, not individual deliveries.
    train = df[df["season_year"] < TEST_START_YEAR].copy()
    test = df[df["season_year"] >= TEST_START_YEAR].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Chronological split produced an empty dataset. "
            "Check the season values and TEST_START_YEAR."
        )

    if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
        raise ValueError(
            "Both train and test must contain labels 0 and 1. "
            "Inspect the seasons or choose a different cutoff."
        )

    # Convert features to numeric values.
    X_train = train[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    X_test = test[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    # Ensure matches do not cross the chronological boundary.
    train_matches = set(train[GROUP])
    test_matches = set(test[GROUP])

    assert train_matches.isdisjoint(test_matches)

    # Baseline predicts the training-set class prior.
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)

    baseline_prob = baseline.predict_proba(X_test)[:, 1]
    baseline_pred = (baseline_prob >= 0.5).astype(int)

    # Train the same model pipeline as the original experiment.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    model_prob = model.predict_proba(X_test)[:, 1]
    model_pred = (model_prob >= 0.5).astype(int)

    baseline_metrics = evaluate(
        y_test, baseline_prob, baseline_pred
    )
    model_metrics = evaluate(
        y_test, model_prob, model_pred
    )

    
    # Calibration analysis for predicted win probabilities.
    # Quantile bins aim to put a similar number of rows in each bin.
    observed_win_rate, mean_predicted_probability = calibration_curve(
        y_test,
        model_prob,
        n_bins=10,
        strategy="quantile",
    )

    calibration_file = (
        ROOT / "data" / "processed" / "calibration_plot.png"
    )
    calibration_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(
        mean_predicted_probability,
        observed_win_rate,
        marker="o",
        label="Logistic regression",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    plt.xlabel("Mean predicted win probability")
    plt.ylabel("Observed win frequency")
    plt.title("CricPulse AI — Probability Calibration")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(calibration_file, dpi=150)
    plt.close()

    print(f"\nCalibration plot saved to: {calibration_file}")


    print("\n=== CRICPULSE AI: CHRONOLOGICAL EVALUATION ===")
    print(f"Training seasons: before {TEST_START_YEAR}")
    print(f"Test seasons: {TEST_START_YEAR} and later")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Training matches: {train[GROUP].nunique():,}")
    print(f"Test matches: {test[GROUP].nunique():,}")
    print(f"Test seasons: {sorted(test['season'].unique())}")
    print(f"Training win rate: {y_train.mean():.4f}")
    print(f"Test win rate: {y_test.mean():.4f}")

    print("\n--- Majority-prior baseline ---")
    for name, value in baseline_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n--- Logistic regression ---")
    for name, value in model_metrics.items():
        print(f"{name}: {value:.4f}")

    # Save a compact report for the portfolio.
    report = pd.DataFrame([
        {"model": "Majority-prior baseline", **baseline_metrics},
        {"model": "Logistic regression", **model_metrics},
    ])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)

    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
