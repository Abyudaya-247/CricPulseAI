
from src.inspect_match import build_delivery_dataframe


def make_sample_match():
    """Create a tiny match with normal, wide, and no-ball deliveries."""
    return {
        "innings": [
            {
                "team": "Team A",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Batter 1",
                                "bowler": "Bowler 1",
                                "runs": {"batter": 1, "total": 1},
                            },
                            {
                                "batter": "Batter 1",
                                "bowler": "Bowler 1",
                                "runs": {"batter": 0, "total": 1},
                                "extras": {"wides": 1},
                            },
                            {
                                "batter": "Batter 2",
                                "bowler": "Bowler 1",
                                "runs": {"batter": 4, "total": 4},
                            },
                            {
                                "batter": "Batter 2",
                                "bowler": "Bowler 1",
                                "runs": {"batter": 0, "total": 1},
                                "extras": {"noballs": 1},
                            },
                            {
                                "batter": "Batter 2",
                                "bowler": "Bowler 1",
                                "runs": {"batter": 0, "total": 0},
                                "wickets": [
                                    {
                                        "player_out": "Batter 2",
                                        "kind": "bowled",
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        ]
    }


def test_total_runs_and_delivery_count():
    df = build_delivery_dataframe(make_sample_match())

    assert len(df) == 5
    assert df["total_runs"].sum() == 7


def test_wide_and_no_ball_are_not_legal_balls():
    df = build_delivery_dataframe(make_sample_match())

    assert df["is_legal_ball"].tolist() == [
        True, False, True, False, True
    ]
    assert df["is_legal_ball"].sum() == 3


def test_cumulative_runs_and_wickets():
    df = build_delivery_dataframe(make_sample_match())

    assert df["cumulative_runs"].tolist() == [1, 2, 6, 7, 7]
    assert df["cumulative_wickets"].tolist() == [0, 0, 0, 0, 1]


def test_cumulative_legal_balls():
    df = build_delivery_dataframe(make_sample_match())

    assert df["legal_balls_bowled"].tolist() == [1, 1, 2, 2, 3]


def test_bowler_runs_and_wickets():
    df = build_delivery_dataframe(make_sample_match())

    assert df["bowler_runs_conceded"].sum() == 7
    assert df["bowler_wickets"].sum() == 1


def test_bowler_economy_calculation():
    df = build_delivery_dataframe(make_sample_match())

    runs_conceded = df["bowler_runs_conceded"].sum()
    legal_balls = int(df["is_legal_ball"].sum())

    economy = runs_conceded * 6 / legal_balls

    assert round(economy, 2) == 14.00

