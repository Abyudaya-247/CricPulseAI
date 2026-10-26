
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "chase_training_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "win_probability_model.joblib"
PLOT_FILE = ROOT / "data" / "processed" / "calibration_plot.png"

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


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

def load_training_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_FILE}\n"
            "Run: python src/build_training_data.py"
        )

    df = pd.read_csv(DATA_FILE, dtype={"season": str})

    required_columns = FEATURES + [TARGET, GROUP]
    missing = sorted(set(required_columns) - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TARGET, GROUP]).copy()

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[FEATURES].isna().all().any():
        raise ValueError("At least one feature column is entirely missing.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The target must contain only 0 and 1.")

    if df[TARGET].nunique() != 2:
        raise ValueError("The dataset must contain both outcome classes.")

    if df[GROUP].nunique() < 2:
        raise ValueError("At least two unique matches are required.")

    return df


# --------------------------------------------------
# 3. Plot calibration results
# --------------------------------------------------

def save_calibration_plot(y_true, probabilities):
    observed, predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.plot(
        predicted,
        observed,
        marker="o",
        linewidth=2,
        label="CricPulse AI",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Mean predicted win probability")
    plt.ylabel("Observed win frequency")
    plt.title("CricPulse AI - Probability Calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()

    print(f"\nCalibration chart saved to: {PLOT_FILE}")

    print("\n--- Probability calibration ---")

    for predicted_probability, observed_frequency in zip(
        predicted, observed
    ):
        print(
            f"Predicted: {predicted_probability:.3f} | "
            f"Observed: {observed_frequency:.3f}"
        )


# --------------------------------------------------
# 4. Train and evaluate
# --------------------------------------------------

def main():
    df = load_training_data()

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    groups = df[GROUP]

    # Hold out entire matches rather than individual deliveries.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_matches = set(groups.iloc[train_indices])
    test_matches = set(groups.iloc[test_indices])

    if not train_matches.isdisjoint(test_matches):
        raise AssertionError("Match leakage detected in train/test split.")

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "Training or test set contains only one class. "
            "Review the split and source data."
        )

    # Baseline: predict the training-set majority class probability.
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)

    baseline_probabilities = baseline.predict_proba(X_test)[:, 1]
    baseline_predictions = (
        baseline_probabilities >= 0.5
    ).astype(int)

    # Logistic regression pipeline.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    model_probabilities = model.predict_proba(X_test)[:, 1]
    model_predictions = (model_probabilities >= 0.5).astype(int)

    # ----------------------------------------------
    # Print dataset and split information
    # ----------------------------------------------

    print("\n=== CRICPULSE AI: MODEL EVALUATION ===")
    print(f"Total rows: {len(df):,}")
    print(f"Unique matches: {groups.nunique():,}")
    print(f"Training rows: {len(train_indices):,}")
    print(f"Test rows: {len(test_indices):,}")
    print(f"Training matches: {len(train_matches):,}")
    print(f"Test matches: {len(test_matches):,}")
    print(f"Training win rate: {y_train.mean():.3f}")
    print(f"Test win rate: {y_test.mean():.3f}")

    # ----------------------------------------------
    # Evaluate baseline and model
    # ----------------------------------------------

    results = {
        "Majority-prior baseline": (
            y_test,
            baseline_predictions,
            baseline_probabilities,
        ),
        "Logistic regression": (
            y_test,
            model_predictions,
            model_probabilities,
        ),
    }

    for name, (actual, predicted, probabilities) in results.items():
        print(f"\n--- {name} ---")
        print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
        print(f"ROC-AUC: {roc_auc_score(actual, probabilities):.4f}")
        print(
            f"Log loss: "
            f"{log_loss(actual, probabilities, labels=[0, 1]):.4f}"
        )
        print(
            f"Brier score: "
            f"{brier_score_loss(actual, probabilities):.4f}"
        )

    # ----------------------------------------------
    # Calibration diagnostics
    # ----------------------------------------------

    save_calibration_plot(y_test, model_probabilities)

    # ----------------------------------------------
    # Save model and feature schema
    # ----------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
        },
        MODEL_FILE,
    )

    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
