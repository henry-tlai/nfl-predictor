# Imports
import glob
import os
import numpy as np
import pandas as pd
# library that turns script into a webpage
import streamlit as st

# Page Settings: browser tab title, and full width of the window
st.set_page_config(page_title="NFL Weekly Predictor", layout="wide")
st.title("NFL Weekly Predictor: 2026")

# Find the saved prediction files (one per week, e.g. 2026_week_06.csv)
files = sorted(glob.glob("data/predictions/2026_week_*.csv"))

# If no files exist yet, show a message rather than crash the page
if not files:
    st.warning("No prediction files found yet.")
    st.stop()

# Map each week number to its file, e.g. {6: "data/predictions/2026_week_06.csv"}
weeks = {int(os.path.basename(f).split("_week_")[1].split(".")[0]): f for f in files}
week_list = sorted(weeks)

# Dropdown to pick a week; it starts on the most recent one
week = st.selectbox("Week", week_list, index=len(week_list) - 1,
                    format_func=lambda w: f"Week {w}")

# Load the chosen week's predictions
latest = pd.read_csv(weeks[week])
st.header(f"Week {week} predictions")

# Our pick = the team with the higher win probability.
# home_win_prob is the chance the HOME team wins, so above 0.5 means we pick home.
latest["Our pick"] = np.where(latest["home_win_prob"] > 0.5,
                              latest["home_team"], latest["away_team"])

# Confidence = the probability of the team we picked (always 50% or higher).
# If we picked the away team, its chance is 1 minus the home team's chance.
latest["Confidence"] = np.where(latest["home_win_prob"] > 0.5,
                                latest["home_win_prob"], 1 - latest["home_win_prob"])

# Show the columns that matter, one row per game
view = latest[["gameday", "away_team", "home_team", "Our pick", "Confidence",
               "vegas_prob", "spread_line"]]

# Generates the table, removing extra row-number columns
st.dataframe(
    view,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Confidence": st.column_config.NumberColumn(format="percent"),
        "vegas_prob": st.column_config.NumberColumn("Vegas home win %", format="percent"),
    },
)

# Scorecard
st.divider()
st.header("Scorecard")

# The sample file only exists on your own computer (it is not uploaded to GitHub),
# so this checkbox never appears on the live site
use_sample = False
if os.path.exists("data/sample/scorecard_sample.csv"):
    use_sample = st.checkbox("Preview with sample data (weeks 1-5 backtest, not the real record)")

# Show the sample file if the box is ticked, otherwise the real scorecard
score_path = "data/sample/scorecard_sample.csv" if use_sample else "data/scorecard.csv"
if use_sample:
    st.warning("Sample data: retroactive picks for testing. This is not the live record.")

# score.py creates the real file once games have finished
if not os.path.exists(score_path):
    st.info("No results yet. The scorecard fills in after the first games finish.")
else:
    sc = pd.read_csv(score_path)
    total = len(sc)

    # Straight-up record: su_correct is True when the team we picked won
    ours = int(sc["su_correct"].sum())

    # Vegas record on the same games: it "picks" the home team if vegas_prob is above 50%
    home_won = sc["home_score"] > sc["away_score"]
    sc["vegas_correct"] = (sc["vegas_prob"] > 0.5) == home_won
    vegas = int(sc["vegas_correct"].sum())

    # Against-the-spread record (pushes are tied games against the line, so they don't count)
    ats_w = int((sc["ats_result"] == "win").sum())
    ats_l = int((sc["ats_result"] == "loss").sum())
    ats_p = int((sc["ats_result"] == "push").sum())
    ats_text = f"{ats_w}-{ats_l}" + (f"-{ats_p}" if ats_p else "")
    ats_pct = f"{ats_w / (ats_w + ats_l):.1%}" if (ats_w + ats_l) else "n/a"

    # Three boxes: our record, Vegas on the same games, and our ATS record
    col1, col2, col3 = st.columns(3)
    col1.metric("Our record", f"{ours}-{total - ours}", f"{ours / total:.1%}", delta_color="off")
    col2.metric("Vegas favorite", f"{vegas}-{total - vegas}", f"{vegas / total:.1%}", delta_color="off")
    col3.metric("Against the spread", ats_text, ats_pct, delta_color="off")
    st.caption("ATS break-even at standard -110 odds is 52.4%. The ATS record is an experiment, "
               "not betting advice.")

    # Week-by-week record
    sc["is_win"] = sc["ats_result"] == "win"
    sc["is_loss"] = sc["ats_result"] == "loss"
    by_week = sc.groupby("week").agg(
        games=("su_correct", "size"),
        our_correct=("su_correct", "sum"),
        vegas_correct=("vegas_correct", "sum"),
        ats_wins=("is_win", "sum"),
        ats_losses=("is_loss", "sum"),
    ).reset_index()
    by_week["our_pct"] = (by_week["our_correct"] / by_week["games"]).round(3)
    st.subheader("By week")
    st.dataframe(by_week, use_container_width=True, hide_index=True,
                 column_config={"our_pct": st.column_config.NumberColumn("Our accuracy", format="percent")})

    # Game-by-game results, newest week first
    sc["Result"] = np.where(sc["su_correct"], "Correct", "Wrong")
    st.subheader("Every game")
    st.dataframe(
        sc.sort_values(["week", "gameday"], ascending=False)[
            ["week", "gameday", "away_team", "home_team", "our_pick", "winner",
             "away_score", "home_score", "Result", "ats_lean", "ats_result"]],
        use_container_width=True, hide_index=True)