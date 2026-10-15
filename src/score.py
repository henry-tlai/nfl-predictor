# Scorecard: compares saved predictions to real results.
# Run from the project folder:  python src/score.py
import glob
import os

import nflreadpy as nfl
import numpy as np
import pandas as pd

SEASON = 2026

# Project folder (one level above src), so paths work no matter where the script is run from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_loss(y, p):
    # Average penalty for probabilities; confident wrong picks cost the most (lower is better)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    # Average squared probability error (lower is better)
    return float(((p - y) ** 2).mean())


def main():
    # 1. Load every saved prediction file (one per week) into one table
    files = sorted(glob.glob(os.path.join(ROOT, "data", "predictions", f"{SEASON}_week_*.csv")))
    if not files:
        print("No prediction files found yet.")
        return
    preds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # 2. Pull final scores and attach them to our predictions by game_id
    games = nfl.load_schedules(seasons=[SEASON]).to_pandas()
    results = games[["game_id", "home_score", "away_score"]]
    df = preds.merge(results, on="game_id", how="left")

    # 3. Keep only games that have finished
    done = df.dropna(subset=["home_score", "away_score"]).copy()
    if done.empty:
        print(f"{len(preds)} predictions saved, but no games have finished yet. Check back after kickoff.")
        return

    # 4. Work out what actually happened
    done["margin"] = done["home_score"] - done["away_score"]  # positive = home team won by that many

    # 5. Straight-up record (ties have no winner, so they are skipped)
    su = done[done["margin"] != 0].copy()
    su["home_win"] = (su["margin"] > 0).astype(int)
    su["our_pick_home"] = su["home_win_prob"] > 0.5
    su["su_correct"] = su["our_pick_home"] == (su["home_win"] == 1)
    y = su["home_win"].to_numpy()

    # 6. Against-the-spread record (a push, where margin equals the spread, is skipped)
    ats = done[done["margin"] != done["spread_line"]].copy()
    ats["home_covers"] = ats["margin"] > ats["spread_line"]
    ats["ats_correct"] = (ats["ats_lean"] == ats["home_team"]) == ats["home_covers"]

   
    # 7. Print the scorecard: our model next to Vegas on the same games
    print(f"Scored {len(su)} finished games across {su['week'].nunique()} week(s)\n")
    print(f"Straight up:  {int(su['su_correct'].sum())}-{int((~su['su_correct']).sum())}"
          f"  ({su['su_correct'].mean():.1%})")
    print(f"Vegas favorite: {((su['vegas_prob'] > 0.5) == (su['home_win'] == 1)).mean():.1%}")
    print(f"Log loss: ours {log_loss(y, su['home_win_prob'].to_numpy()):.3f}"
          f" | Vegas {log_loss(y, su['vegas_prob'].to_numpy()):.3f}")
    print(f"Brier:    ours {brier(y, su['home_win_prob'].to_numpy()):.3f}"
          f" | Vegas {brier(y, su['vegas_prob'].to_numpy()):.3f}")
    print(f"Against the spread: {int(ats['ats_correct'].sum())}-{int((~ats['ats_correct']).sum())}"
          f"  ({ats['ats_correct'].mean():.1%}, break-even is 52.4%)\n")

    # 8. Week-by-week record
    by_week = su.groupby("week")["su_correct"].agg(["sum", "count"])
    by_week.columns = ["correct", "games"]
    print(by_week)

    # 9. Save game-by-game results for the dashboard (ties dropped: no winner to score)
    out = done[done["margin"] != 0].copy()
    out["our_pick"] = np.where(out["home_win_prob"] > 0.5, out["home_team"], out["away_team"])
    out["winner"] = np.where(out["margin"] > 0, out["home_team"], out["away_team"])
    out["su_correct"] = out["our_pick"] == out["winner"]  # did the team we picked win?

    # ATS result: did the side we leaned toward cover the spread? (push = margin equals the spread)
    home_covers = out["margin"] > out["spread_line"]
    lean_home = out["ats_lean"] == out["home_team"]
    out["ats_result"] = np.where(out["margin"] == out["spread_line"], "push",
                                 np.where(lean_home == home_covers, "win", "loss"))

    cols = ["week", "gameday", "away_team", "home_team", "our_pick", "winner",
            "home_win_prob", "vegas_prob", "spread_line", "ats_lean", "ats_result",
            "home_score", "away_score", "su_correct"]
    out[cols].to_csv(os.path.join(ROOT, "data", "scorecard.csv"), index=False)
    print("\nSaved game-by-game results to data/scorecard.csv")

if __name__ == "__main__":
    main()