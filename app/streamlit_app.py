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

# score.py creates this file once games have finished
if not os.path.exists("data/scorecard.csv"):
    st.info("No results yet. The scorecard fills in after the first games finish.")
else:
    sc = pd.read_csv("data/scorecard.csv")
    total = len(sc)

    # Our record: su_correct is True when the team we picked won
    ours = int(sc["su_correct"].sum())

    # Vegas record on the same games: it "picks" the home team if vegas_prob is above 50%
    home_won = sc["home_score"] > sc["away_score"]
    vegas = int(((sc["vegas_prob"] > 0.5) == home_won).sum())

    # Two side-by-side boxes showing wins-losses and the percentage
    col1, col2 = st.columns(2)
    col1.metric("Our record", f"{ours}-{total - ours}", f"{ours / total:.1%}", delta_color="off")
    col2.metric("Vegas favorite", f"{vegas}-{total - vegas}", f"{vegas / total:.1%}", delta_color="off")

    # Game-by-game results underneath
    st.dataframe(sc, use_container_width=True, hide_index=True)