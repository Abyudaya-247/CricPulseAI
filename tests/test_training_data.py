
from src.build_training_data import build_match_rows


def make_match(winner="Team B", extra_info=None):
    info = {
        "match_type": "T20",
        "overs": 20,
        "teams": ["Team A", "Team B"],
        "season": 2020,
        "outcome": {"winner": winner},
    }

    if extra_info:
        info.update(extra_info)

    return {
        "info": info,
        "innings": [
            {
                "team": "Team A",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "A1",
                                "bowler": "B1",
                                "runs": {"batter": 4, "total": 4},
                            }
                        ],
                    }
                ],
            },
            {
                "team": "Team B",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "B1",
                                "bowler": "A1",
                                "runs": {"batter": 1, "total": 1},
                            },
                            {
                                "batter": "B1",
                                "bowler": "A1",
                                "runs": {"batter": 4, "total": 4},
                            },
                        ],
                    }
                ],
            },
        ],
    }


def test_completed_chase_generates_rows():
    rows = build_match_rows(make_match(), "test_match")

    assert len(rows) == 2
    assert rows[0]["match_id"] == "test_match"
    assert rows[0]["target_runs"] == 5


def test_features_describe_state_before_delivery():
    rows = build_match_rows(make_match(), "test_match")

    assert rows[0]["current_score"] == 0
    assert rows[0]["runs_required"] == 5
    assert rows[0]["balls_remaining"] == 120

    assert rows[1]["current_score"] == 1
    assert rows[1]["runs_required"] == 4
    assert rows[1]["balls_remaining"] == 119


def test_winner_label_is_consistent():
    rows = build_match_rows(
        make_match(winner="Team B"),
        "test_match",
    )

    assert len(rows) > 0
    assert all(row["chase_win"] == 1 for row in rows)


def test_dls_match_is_excluded():
    match = make_match(
        extra_info={
            "outcome": {
                "winner": "Team B",
                "method": "D/L",
            }
        }
    )

    assert build_match_rows(match, "dls_match") == []


def test_tied_match_without_winner_is_excluded():
    match = make_match(
        winner=None,
        extra_info={"outcome": {"result": "tie"}},
    )

    assert build_match_rows(match, "tie_match") == []


def test_super_over_match_is_excluded():
    match = make_match()

    match["innings"].append({
        "team": "Team A",
        "super_over": True,
        "overs": [],
    })

    assert build_match_rows(match, "super_over_match") == []

