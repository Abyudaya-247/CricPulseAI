
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
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 18px 20px;
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    h2 {
        padding-top: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding: 0.5rem 0 1rem 0;">
        <h1 style="margin-bottom: 0.2rem;">🏏 CricPulse AI</h1>
        <p style="color: #8b949e; font-size: 1.05rem; margin-top: 0;">
            Cricket intelligence through data analytics and machine learning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


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

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.metric(
        label="Total Runs",
        value=f"{total_runs:,}",
        help="Runs scored across the selected innings.",
    )

with col2:
    st.metric(
        label="Wickets Lost",
        value=f"{total_wickets:,}",
        help="Total wickets recorded in the selected deliveries.",
    )

with col3:
    st.metric(
        label="Deliveries",
        value=f"{total_deliveries:,}",
        help="Recorded delivery rows, including extras.",
    )

with col4:
    st.metric(
        label="Innings Selected",
        value=f"{innings_count:,}",
        help="Number of distinct innings in the selection.",
    )


summary_display = summary[
    ["innings", "batting_team", "runs", "wickets", "overs"]
].rename(
    columns={
        "innings": "Innings",
        "batting_team": "Batting Team",
        "runs": "Runs",
        "wickets": "Wickets",
        "overs": "Overs",
    }
)

st.dataframe(
    summary_display,
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

fig.update_layout(
    template="plotly_white",
    title=dict(
        text="Runs Scored Throughout Each Innings",
        x=0,
        xanchor="left",
    ),
    showlegend=True,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=100, b=20),
    height=460,
    legend=dict(
        title_text="",
        orientation="h",
        yanchor="bottom",
        y=1.08,
        xanchor="left",
        x=0,
    ),
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(
    title_text="Cumulative Runs",
    rangemode="tozero",
    gridcolor="rgba(128, 128, 128, 0.15)",
)

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
        "batting_team": "Batting Team",
        "wickets": "Wickets lost",
    },
    title="Runs Scored per Over",
    template="plotly_white",
)

fig.update_layout(
    template="plotly_white",
    title=dict(
        text="Runs Scored Throughout Each Innings",
        x=0,
        xanchor="left",
    ),
    showlegend=True,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=100, b=20),
    height=460,
    legend=dict(
        title_text="",
        orientation="h",
        yanchor="bottom",
        y=1.08,
        xanchor="left",
        x=0,
    ),
)

fig_over.update_xaxes(
    showgrid=False,
    title_text="Over Number",
)

fig_over.update_yaxes(
    title_text="Runs Scored",
    rangemode="tozero",
    gridcolor="rgba(128, 128, 128, 0.15)",
)

st.plotly_chart(fig_over, use_container_width=True)


st.dataframe(
    over_stats.sort_values(["innings", "over"]),
    use_container_width=True,
    hide_index=True,
)





# -----------------------------
# AI Win Probability Predictor
# -----------------------------
st.divider()
st.header("🤖 AI Win Probability Predictor")

st.caption(
    "Estimate the chasing team's chance of winning using a model "
    "trained on historical IPL chase situations. This is an experimental "
    "estimate, not a guarantee or live match prediction."
)

MODEL_FILE = PROJECT_ROOT / "models" / "win_probability_model.joblib"

try:
    if not MODEL_FILE.exists():
        st.error(
            "Trained model not found. Ensure "
            "models/win_probability_model.joblib exists."
        )
    else:
        artifact = joblib.load(MODEL_FILE)
        model = artifact["model"]
        features = artifact["features"]

        with st.form("win_probability_form"):
            st.subheader("Chase Situation")

            input_col1, input_col2 = st.columns(2)

            with input_col1:
                target_runs = st.number_input(
                    "Target score",
                    min_value=1,
                    max_value=500,
                    value=180,
                    step=1,
                    help="The total score the chasing team needs to reach.",
                )

                current_score = st.number_input(
                    "Current score",
                    min_value=0,
                    max_value=500,
                    value=75,
                    step=1,
                )

                wickets_lost = st.number_input(
                    "Wickets lost",
                    min_value=0,
                    max_value=10,
                    value=2,
                    step=1,
                )

            with input_col2:
                balls_bowled = st.number_input(
                    "Legal balls bowled",
                    min_value=0,
                    max_value=120,
                    value=60,
                    step=1,
                    help="For a standard 20-over innings, the maximum is 120.",
                )

            submitted = st.form_submit_button(
                "Predict Win Probability",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            runs_required = max(int(target_runs) - int(current_score), 0)
            balls_remaining = 120 - int(balls_bowled)
            wickets_remaining = 10 - int(wickets_lost)

            if current_score >= target_runs:
                st.success(
                    "The target has already been reached. "
                    "The chase is complete."
                )

            elif wickets_remaining == 0:
                st.warning(
                    "All 10 wickets have fallen. "
                    "The chasing innings is complete."
                )

            elif balls_remaining == 0:
                st.warning(
                    "The 20-over limit has been reached. "
                    "The chase is complete."
                )

            else:
                current_run_rate = (
                    int(current_score) * 6 / int(balls_bowled)
                    if balls_bowled > 0
                    else 0.0
                )

                required_run_rate = (
                    runs_required * 6 / balls_remaining
                )

                input_values = {
                    "target_runs": float(target_runs),
                    "current_score": float(current_score),
                    "runs_required": float(runs_required),
                    "wickets_remaining": float(wickets_remaining),
                    "balls_remaining": float(balls_remaining),
                    "current_run_rate": float(current_run_rate),
                    "required_run_rate": float(required_run_rate),
                }

                input_row = pd.DataFrame(
                    [[input_values[name] for name in features]],
                    columns=features,
                )

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_row)[0]
                    classes = list(model.classes_)

                    if 1 not in classes:
                        st.error(
                            "The trained model does not contain "
                            "the expected win class (1)."
                        )
                    else:
                        win_probability = float(
                            probabilities[classes.index(1)]
                        )
                        win_probability = min(
                            max(win_probability, 0.0), 1.0
                        )

                        st.subheader("Prediction Result")

                        result_col1, result_col2 = st.columns(2)

                        result_col1.metric(
                            "Estimated Chance of Winning",
                            f"{win_probability:.1%}",
                        )

                        result_col2.metric(
                            "Estimated Chance of Losing",
                            f"{1 - win_probability:.1%}",
                        )

                        st.progress(win_probability)

                        st.caption(
                            f"{runs_required} runs needed from "
                            f"{balls_remaining} legal balls, with "
                            f"{wickets_remaining} wickets remaining."
                        )

                        st.info(
                            "This is a model-based estimate from "
                            "historical data. It is not a live forecast, "
                            "and its calibration and reliability may vary."
                        )
                else:
                    st.error(
                        "The loaded model does not support "
                        "probability predictions."
                    )

except Exception as exc:
    st.error(f"Could not load or run the prediction model: {exc}")


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "CricPulse AI • Historical cricket analytics • "
    "Experimental machine-learning win probability"
)

