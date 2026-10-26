
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_FILE = RAW_DIR / "sample_match.json"


def load_match(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)



def build_delivery_dataframe(match):
    rows = []

    for innings_number, innings in enumerate(
        match.get("innings", []), start=1
    ):
        batting_team = innings.get("team", "")
        cumulative_runs = 0
        cumulative_wickets = 0
        legal_balls = 0

        for over in innings.get("overs", []):
            over_number = over.get("over")

            for delivery_index, delivery in enumerate(
                over.get("deliveries", []), start=1
            ):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                total_runs = runs.get("total", 0)
                batter_runs = runs.get("batter", 0)

                wides = extras.get("wides", 0)
                no_balls = extras.get("noballs", 0)
                byes = extras.get("byes", 0)
                legbyes = extras.get("legbyes", 0)
                penalty = extras.get("penalty", 0)

                is_wide = "wides" in extras
                is_no_ball = "noballs" in extras
                is_legal_ball = not (is_wide or is_no_ball)

                # A wide is not a ball faced by the batter.
                # A no-ball counts as a ball faced.
                balls_faced = 0 if is_wide else 1

                # Runs charged to the bowler exclude byes,
                # leg-byes and penalty runs.
                bowler_runs = (
                    total_runs - byes - legbyes - penalty
                )

                wicket_types = [
                    wicket.get("kind", "")
                    for wicket in wickets
                ]

                # These dismissal types are not credited
                # to the bowler.
                uncredited_types = {
                    "run out",
                    "retired hurt",
                    "retired out",
                    "obstructing the field",
                    "timed out",
                }

                bowler_wickets = sum(
                    kind not in uncredited_types
                    for kind in wicket_types
                )

                cumulative_runs += total_runs
                cumulative_wickets += len(wickets)

                if is_legal_ball:
                    legal_balls += 1

                rows.append({
                    "innings": innings_number,
                    "batting_team": batting_team,
                    "over": over_number,
                    "delivery_index_in_over": delivery_index,
                    "batter": delivery.get("batter"),
                    "bowler": delivery.get("bowler"),
                    "batter_runs": batter_runs,
                    "total_runs": total_runs,
                    "extras": sum(extras.values()),
                    "wides": wides,
                    "noballs": no_balls,
                    "byes": byes,
                    "legbyes": legbyes,
                    "penalty_runs": penalty,
                    "is_wide": is_wide,
                    "is_no_ball": is_no_ball,
                    "balls_faced": balls_faced,
                    "bowler_runs_conceded": bowler_runs,
                    "wickets": len(wickets),
                    "bowler_wickets": bowler_wickets,
                    "wicket_types": ", ".join(wicket_types),
                    "is_legal_ball": is_legal_ball,
                    "cumulative_runs": cumulative_runs,
                    "cumulative_wickets": cumulative_wickets,
                    "legal_balls_bowled": legal_balls,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("The match contains no recorded deliveries.")

    return df



def main():
    if not SAMPLE_FILE.exists():
        print(f"Sample file not found: {SAMPLE_FILE}")
        sys.exit(1)

    match = load_match(SAMPLE_FILE)
    info = match.get("info", {})
    df = build_delivery_dataframe(match)

    print("\n=== CRICPULSE AI ===")
    print("Teams:", info.get("teams", []))
    print("Date:", info.get("dates", []))
    print("Venue:", info.get("venue", "Unknown"))

    print("\n=== FIRST 10 DELIVERIES ===")
    print(df.head(10).to_string(index=False))

    summary = (
        df.groupby(["innings", "batting_team"], as_index=False)
        .agg(
            total_runs=("total_runs", "sum"),
            wickets=("wickets", "sum"),
            recorded_deliveries=("total_runs", "size"),
            legal_balls=("is_legal_ball", "sum"),
        )
    )

    print("\n=== INNINGS SUMMARY ===")
    print(summary.to_string(index=False))

    # Validate basic score consistency.
    assert df["total_runs"].ge(0).all()
    assert df["batter_runs"].ge(0).all()
    assert df["wickets"].ge(0).all()
    assert df["is_legal_ball"].isin([True, False]).all()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    deliveries_path = PROCESSED_DIR / "sample_deliveries.csv"
    summary_path = PROCESSED_DIR / "innings_summary.csv"

    df.to_csv(deliveries_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nValidation passed.")
    print(f"Deliveries saved to: {deliveries_path}")
    print(f"Innings summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
