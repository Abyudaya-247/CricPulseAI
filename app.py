
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import joblib


# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "sample_deliveries.csv"

st.set_page_config(
    page_title="CricPulse AI",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 CricPulse AI")
st.caption("Historical Cricket Analytics | First Prototype")


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    required_columns = {
        "innings",
        "batting_team",
        "over",
        "batter",
        "bowler",
        "batter_runs",
        "total_runs",
        "wickets",
        "cumulative_runs",
        "cumulative_wickets",
        "is_legal_ball",
        "legal_balls_bowled",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info("Run `python src/inspect_match.py` first.")
    st.stop()

try:
    df = load_data(str(DATA_FILE))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if df.empty:
    st.warning("The dataset contains no deliveries.")
    st.stop()


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Match Filters")

innings_options = sorted(df["innings"].dropna().unique().tolist())

selected_innings = st.sidebar.multiselect(
    "Select innings",
    options=innings_options,
    default=innings_options,
)

filtered = df[df["innings"].isin(selected_innings)].copy()

if filtered.empty:
    st.warning("Select at least one innings.")
    st.stop()


# -----------------------------
# Match summary
# -----------------------------
st.subheader("Match Overview")

summary = (
    filtered.groupby(["innings", "batting_team"], as_index=False)
    .agg(
        runs=("total_runs", "sum"),
        wickets=("wickets", "sum"),
        legal_balls=("is_legal_ball", "sum"),
    )
)

summary["overs"] = summary["legal_balls"].apply(
    lambda balls: f"{int(balls) // 6}.{int(balls) % 6}"
)

total_runs = int(filtered["total_runs"].sum())
total_wickets = int(filtered["wickets"].sum())
total_deliveries = len(filtered)
innings_count = filtered["innings"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Runs in selection", f"{total_runs:,}")
col2.metric("Wickets", total_wickets)
col3.metric("Recorded deliveries", f"{total_deliveries:,}")
col4.metric("Innings selected", innings_count)

st.dataframe(
    summary[
        ["innings", "batting_team", "runs", "wickets", "overs"]
    ],
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# Scoring progression
# -----------------------------
st.subheader("Scoring Progression")

chart_data = filtered.sort_values(
    ["innings", "legal_balls_bowled"]
).copy()

chart_data["ball_sequence"] = (
    chart_data.groupby("innings").cumcount() + 1
)

fig = px.line(
    chart_data,
    x="ball_sequence",
    y="cumulative_runs",
    color="batting_team",
    line_group="innings",
    markers=False,
    hover_data=[
        "innings",
        "over",
        "batter",
        "bowler",
        "total_runs",
        "cumulative_wickets",
    ],
    labels={
        "ball_sequence": "Recorded delivery number",
        "cumulative_runs": "Cumulative runs",
        "batting_team": "Batting team",
        "cumulative_wickets": "Wickets lost",
    },
    title="Runs scored throughout each innings",
)

fig.update_layout(legend_title_text="Batting team")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Batter performance
# -----------------------------
st.subheader("Batter Performance")

batter_stats = (
    filtered.groupby(
        ["innings", "batting_team", "batter"],
        as_index=False,
    )
    .agg(
        runs=("batter_runs", "sum"),
        balls_faced=("balls_faced", "sum"),
        fours=("batter_runs", lambda s: (s == 4).sum()),
        sixes=("batter_runs", lambda s: (s == 6).sum()),
    )
)

batter_stats["strike_rate"] = (
    batter_stats["runs"] * 100
    / batter_stats["balls_faced"].replace(0, float("nan"))
)

st.dataframe(
    batter_stats.sort_values("runs", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Strike rate = batter runs / balls faced × 100. "
    "Wides are excluded from balls faced; no-balls are included."
)


# -----------------------------
# Bowler performance
# -----------------------------
st.subheader("Bowler Performance")

required_bowling_columns = {
    "bowler_runs_conceded",
    "bowler_wickets",
    "is_legal_ball",
    "innings",
    "batting_team",
    "bowler",
}

if required_bowling_columns.issubset(filtered.columns):
    bowling_stats = (
        filtered.groupby(
            ["innings", "batting_team", "bowler"],
            as_index=False,
        )
        .agg(
            runs_conceded=("bowler_runs_conceded", "sum"),
            wickets=("bowler_wickets", "sum"),
            legal_balls=("is_legal_ball", "sum"),
        )
    )

    bowling_stats["overs"] = bowling_stats["legal_balls"].apply(
        lambda balls: f"{int(balls) // 6}.{int(balls) % 6}"
    )

    bowling_stats["economy"] = (
        bowling_stats["runs_conceded"] * 6
        / bowling_stats["legal_balls"].replace(0, float("nan"))
    ).round(2)

    st.dataframe(
        bowling_stats[
            [
                "innings",
                "batting_team",
                "bowler",
                "overs",
                "runs_conceded",
                "wickets",
                "economy",
            ]
        ].sort_values(
            ["wickets", "economy"],
            ascending=[False, True],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Economy = runs conceded per six legal deliveries. "
        "The batting-team column identifies the opposition faced."
    )
else:
    st.warning(
        "Bowling fields are missing. Run "
        "`python src/inspect_match.py` to regenerate the processed data."
    )


# -----------------------------
# Over-by-over analysis
# -----------------------------
st.subheader("Over-by-Over Analysis")

over_stats = (
    filtered.groupby(
        ["innings", "batting_team", "over"],
        as_index=False,
    )
    .agg(
        runs=("total_runs", "sum"),
        wickets=("wickets", "sum"),
    )
)

fig_over = px.bar(
    over_stats,
    x="over",
    y="runs",
    color="batting_team",
    barmode="group",
    hover_data=["innings", "wickets"],
    labels={
        "over": "Over number (zero-based in source data)",
        "runs": "Runs scored",
        "batting_team": "Batting team",
        "wickets": "Wickets lost",
    },
    title="Runs scored per over",
)

st.plotly_chart(fig_over, use_container_width=True)

st.dataframe(
    over_stats.sort_values(["innings", "over"]),
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# AI win-probability predictor
# -----------------------------
st.divider()
st.header("🤖 AI Win Probability Predictor")
st.caption(
    "Experimental model trained on historical IPL chase situations. "
    "Predictions are estimates, not guarantees."
)

MODEL_FILE = PROJECT_ROOT / "models" / "win_probability_model.joblib"

if not MODEL_FILE.exists():
    st.warning(
        "Trained model not found. Run "
        "`python src/train_model.py` first."
    )
else:
    try:
        artifact = joblib.load(MODEL_FILE)
        model = artifact["model"]
        features = artifact["features"]

        with st.form("win_probability_form"):
            st.subheader("Enter the match situation")

            col1, col2 = st.columns(2)

            with col1:
                target = st.number_input(
                    "Target score",
                    min_value=1,
                    max_value=400,
                    value=180,
                    step=1,
                )

                current_score = st.number_input(
                    "Current score",
                    min_value=0,
                    max_value=400,
                    value=96,
                    step=1,
                )

                wickets_lost = st.slider(
                    "Wickets lost",
                    min_value=0,
                    max_value=10,
                    value=3,
                )

            with col2:
                legal_balls = st.slider(
                    "Legal balls bowled",
                    min_value=0,
                    max_value=120,
                    value=72,
                )

            submitted = st.form_submit_button(
                "Predict win probability",
                type="primary",
            )

        if submitted:
            if current_score >= target:
                st.info(
                    "The target has already been reached. "
                    "This is a completed chase state, not a "
                    "prediction problem."
                )

            elif wickets_lost >= 10 or legal_balls >= 120:
                st.info(
                    "The innings has ended. Enter a non-terminal "
                    "match situation."
                )

            else:
                runs_required = target - current_score
                balls_remaining = 120 - legal_balls

                current_run_rate = (
                    current_score * 6 / legal_balls
                    if legal_balls > 0
                    else 0.0
                )

                required_run_rate = (
                    runs_required * 6 / balls_remaining
                )

                input_row = pd.DataFrame([{
                    "target_runs": target,
                    "current_score": current_score,
                    "runs_required": runs_required,
                    "wickets_remaining": 10 - wickets_lost,
                    "balls_remaining": balls_remaining,
                    "current_run_rate": current_run_rate,
                    "required_run_rate": required_run_rate,
                }])[features]

                probability = model.predict_proba(input_row)[0]
                classes = list(model.classes_)
                win_probability = float(
                    probability[classes.index(1)]
                )

                st.subheader("Prediction")

                col1, col2, col3 = st.columns(3)

                col1.metric("Runs required", runs_required)
                col2.metric("Balls remaining", balls_remaining)
                col3.metric(
                    "Required run rate",
                    f"{required_run_rate:.2f}",
                )

                st.progress(win_probability)

                st.metric(
                    "Estimated chasing-team win probability",
                    f"{win_probability:.1%}",
                )

                st.caption(
                    "This is the model's raw probability estimate. "
                    "It has not yet been independently validated for "
                    "calibration across different match situations."
                )

    except Exception as exc:
        st.error(f"Could not load or run the prediction model: {exc}")




st.divider()
st.caption(
    "CricPulse AI • Historical data prototype • "
    "Live scores and win-probability predictions are future milestones."
)
