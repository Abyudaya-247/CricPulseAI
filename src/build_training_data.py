
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ipl_json"
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "chase_training_data.csv"
)

MAX_OVERS = 20
MAX_BALLS = MAX_OVERS * 6


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def innings_total(innings):
    return sum(
        delivery.get("runs", {}).get("total", 0)
        for over in innings.get("overs", [])
        for delivery in over.get("deliveries", [])
    )


def is_legal_delivery(delivery):
    extras = delivery.get("extras", {})
    return "wides" not in extras and "noballs" not in extras


def wickets_on_delivery(delivery):
    # Retired hurt is not a completed dismissal.
    return sum(
        wicket.get("kind") != "retired hurt"
        for wicket in delivery.get("wickets", [])
    )


def build_match_rows(match, match_id):
    info = match.get("info", {})
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")

    # Restrict the first model to standard IPL T20 matches.
    if info.get("match_type") != "T20":
        return []

    if info.get("overs") != MAX_OVERS:
        return []

    if not winner or outcome.get("result"):
        return []

    # Exclude shortened/DLS matches in this first version.
    if outcome.get("method") == "D/L":
        return []

    
    all_innings = match.get("innings", [])

    # Exclude the entire match if it contains a Super Over.
    # Otherwise, a Super Over winner could be incorrectly used
    # to label a tied regulation chase.
    if any(item.get("super_over", False) for item in all_innings):
        return []

    innings = all_innings

    if len(innings) != 2:
        return []

    first_innings, chase_innings = innings

    if len(innings) != 2:
        return []

    first_innings, chase_innings = innings
    chasing_team = chase_innings.get("team")

    if not chasing_team or winner not in info.get("teams", []):
        return []

    target = innings_total(first_innings) + 1

    if target <= 1:
        return []

    rows = []
    score = 0
    wickets_lost = 0
    legal_balls = 0

    for over in chase_innings.get("overs", []):
        for delivery in over.get("deliveries", []):
            balls_remaining = max(0, MAX_BALLS - legal_balls)
            runs_required = max(0, target - score)

            # Record the state BEFORE the next delivery.
            if runs_required > 0 and balls_remaining > 0:
                current_rr = (
                    score * 6 / legal_balls
                    if legal_balls > 0 else 0.0
                )

                required_rr = (
                    runs_required * 6 / balls_remaining
                )

                rows.append({
                    "match_id": match_id,
                    "season": str(info.get("season", "unknown")),
                    "chasing_team": chasing_team,
                    "opponent": first_innings.get("team"),
                    "target_runs": target,
                    "current_score": score,
                    "runs_required": runs_required,
                    "wickets_lost": wickets_lost,
                    "wickets_remaining": max(0, 10 - wickets_lost),
                    "legal_balls_bowled": legal_balls,
                    "balls_remaining": balls_remaining,
                    "current_run_rate": current_rr,
                    "required_run_rate": required_rr,
                    "chase_win": int(winner == chasing_team),
                })

            # Update the state using this delivery.
            score += delivery.get("runs", {}).get("total", 0)
            wickets_lost += wickets_on_delivery(delivery)

            if is_legal_delivery(delivery):
                legal_balls += 1

            # Do not record states after the chase is completed.
            if score >= target or wickets_lost >= 10:
                break

        if score >= target or wickets_lost >= 10:
            break

    return rows


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"IPL JSON directory not found: {RAW_DIR}"
        )

    files = sorted(RAW_DIR.rglob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No JSON files found inside {RAW_DIR}"
        )

    all_rows = []
    eligible_matches = 0
    skipped_or_ineligible = 0

    for path in files:
        try:
            match = load_json(path)
            rows = build_match_rows(match, path.stem)

            if rows:
                all_rows.extend(rows)
                eligible_matches += 1
            else:
                skipped_or_ineligible += 1

        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            print(f"Skipping {path.name}: {exc}")
            skipped_or_ineligible += 1

    if not all_rows:
        raise ValueError(
            "No eligible chase states were generated. "
            "Check the archive format and match metadata."
        )

    df = pd.DataFrame(all_rows)

    # Basic dataset quality checks.
    assert df["match_id"].notna().all()
    assert df["chase_win"].isin([0, 1]).all()
    assert df["runs_required"].ge(1).all()
    assert df["balls_remaining"].between(1, MAX_BALLS).all()
    assert df["wickets_remaining"].between(0, 10).all()
    assert df["match_id"].nunique() == eligible_matches

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n=== CRICPULSE AI: TRAINING DATA ===")
    print(f"JSON files scanned: {len(files)}")
    print(f"Eligible matches: {eligible_matches}")
    print(f"Skipped/ineligible files: {skipped_or_ineligible}")
    print(f"Training rows: {len(df):,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")
    print("\nOutcome distribution:")
    print(df["chase_win"].value_counts().sort_index())
    print("\nSample rows:")
    print(df.head().to_string(index=False))
    print(f"\nSaved dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
