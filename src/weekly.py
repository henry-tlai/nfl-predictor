# Weekly NFL predictor: loads fresh data, trains the models, and saves next week's predictions.
# Run from the project folder:  python src/weekly.py
# If that week's file already exists it is NOT overwritten, so predictions made
# before kickoff stay untouched. Add --force to rebuild it anyway.
import os
import sys
from datetime import datetime

import nflreadpy as nfl
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from elo_baseline import run_elo, vegas_prob, TEAM_FIX

SEASON = 2026
FIRST_SEASON = 2015
FEATS = ["elo_diff", "off_diff", "def_diff"]  # same inputs as the notebook models

# Project folder (one level above src), so paths work no matter where the script is run from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    force = "--force" in sys.argv

    # 1. Load fresh data: one row per game, and one row per team per game
    games = nfl.load_schedules(seasons=range(FIRST_SEASON, SEASON + 1)).to_pandas()
    stats = nfl.load_team_stats(seasons=range(FIRST_SEASON, SEASON + 1)).to_pandas()

    # 2. Elo ratings, built game by game in date order
    df, _ = run_elo(games)

    # 3. EPA per play for offense, and for defense (the opponent's offense in the same game)
    s = stats.copy()
    plays = s["attempts"] + s["carries"] + s["sacks_suffered"]
    s["off_epa_pp"] = (s["passing_epa"] + s["rushing_epa"]) / plays
    opp = s[["game_id", "team", "off_epa_pp"]].rename(
        columns={"team": "opponent_team", "off_epa_pp": "def_epa_pp"})
    s = s.merge(opp, on=["game_id", "opponent_team"], how="left")

    # 4. Rolling average of the previous 6 games; shift(1) keeps each game out of its own average
    s = s.sort_values(["team", "season", "week"])
    for col in ["off_epa_pp", "def_epa_pp"]:
        s[col + "_r6"] = (s.groupby("team")[col]
                           .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean()))
    s["team"] = s["team"].replace(TEAM_FIX)  # same team abbreviations as the Elo table

    # 5. One row per game with Elo plus both teams' rolling features, and the home-minus-away gaps
    home = s[["game_id", "team", "off_epa_pp_r6", "def_epa_pp_r6"]].rename(
        columns={"team": "home_team", "off_epa_pp_r6": "home_off", "def_epa_pp_r6": "home_def"})
    away = s[["game_id", "team", "off_epa_pp_r6", "def_epa_pp_r6"]].rename(
        columns={"team": "away_team", "off_epa_pp_r6": "away_off", "def_epa_pp_r6": "away_def"})
    data = (df.merge(home, on=["game_id", "home_team"], how="left")
              .merge(away, on=["game_id", "away_team"], how="left"))
    data["elo_diff"] = data["elo_home"] - data["elo_away"]
    data["off_diff"] = data["home_off"] - data["away_off"]
    data["def_diff"] = data["home_def"] - data["away_def"]

    # 6. Train on every finished game so far: win model (ties dropped) and margin model
    done_games = data.dropna(subset=FEATS + ["home_score", "away_score"]).copy()
    win_train = done_games[done_games["home_score"] != done_games["away_score"]].copy()
    win_train["home_win"] = (win_train["home_score"] > win_train["away_score"]).astype(int)
    win_model = LogisticRegression().fit(win_train[FEATS], win_train["home_win"])

    done_games["margin"] = done_games["home_score"] - done_games["away_score"]
    margin_model = Ridge(alpha=1.0).fit(done_games[FEATS], done_games["margin"])

    # 7. Each team's current form = average of its last 6 finished games.
    # There is no shift(1) here because the game being predicted hasn't happened yet.
    finished = s.dropna(subset=["off_epa_pp", "def_epa_pp"])
    form = (finished.groupby("team").tail(6)
                    .groupby("team")[["off_epa_pp", "def_epa_pp"]].mean())

    # 8. Find next week's games: no score yet AND a date that hasn't passed.
    # (Last night's game may still be missing its score in the data, so skip it.)
    today = pd.Timestamp.today().normalize()
    upcoming = df[df["home_score"].isna() & (df["season"] == SEASON)].copy()
    upcoming = upcoming[pd.to_datetime(upcoming["gameday"]) >= today]
    if upcoming.empty:
        print("No upcoming games found. Nothing to predict.")
        return
    next_week = int(upcoming["week"].min())
    nxt = upcoming[upcoming["week"] == next_week].copy()

    # Don't overwrite a week that was already predicted (unless --force)
    out_dir = os.path.join(ROOT, "data", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{SEASON}_week_{next_week:02d}.csv")
    if os.path.exists(path) and not force:
        print(f"Week {next_week} already exists at {path}. Use --force to rebuild it.")
        return

    # 9. Build the same gaps for the upcoming games, then predict
    nxt = nxt.join(form.add_prefix("home_"), on="home_team")
    nxt = nxt.join(form.add_prefix("away_"), on="away_team")
    nxt["elo_diff"] = nxt["elo_home"] - nxt["elo_away"]  # Elo ratings are already current
    nxt["off_diff"] = nxt["home_off_epa_pp"] - nxt["away_off_epa_pp"]
    nxt["def_diff"] = nxt["home_def_epa_pp"] - nxt["away_def_epa_pp"]

    nxt["home_win_prob"] = win_model.predict_proba(nxt[FEATS])[:, 1]  # P(home team wins)
    nxt["vegas_prob"] = vegas_prob(nxt["spread_line"])
    nxt["pred_margin"] = margin_model.predict(nxt[FEATS])
    nxt["edge"] = nxt["pred_margin"] - nxt["spread_line"]  # how much we like home more than Vegas
    nxt["ats_lean"] = nxt["home_team"].where(nxt["edge"] > 0, nxt["away_team"])
    nxt["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 10. Save one file per week
    cols = ["game_id", "season", "week", "gameday", "away_team", "home_team",
            "home_win_prob", "vegas_prob", "spread_line",
            "pred_margin", "edge", "ats_lean", "predicted_at"]
    nxt[cols].to_csv(path, index=False)
    print(f"Saved {len(nxt)} predictions for week {next_week} to {path}")


if __name__ == "__main__":
    main()