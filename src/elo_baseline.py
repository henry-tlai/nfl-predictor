"""Elo baseline + walk-forward evaluation for NFL straight-up predictions.

Idea: every team has a rating. Before each game we turn the rating gap into a
win probability, then after the game we nudge both ratings based on the result.
Because we always predict BEFORE updating, the model never sees the future.
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss

# Teams that moved cities have old abbreviations in older seasons.
# Mapping them to the current name keeps one team's history together.
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def expected_home_win(home_elo, away_elo, hfa, neutral=False):
    """Turn a rating gap into the home team's win probability."""
    # Home teams get a bonus (hfa) unless the game is at a neutral site
    diff = home_elo - away_elo + (0.0 if neutral else hfa)
    # Standard Elo formula: a 400-point gap means roughly 10-to-1 odds
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def run_elo(games, k=20.0, hfa=30.0, regress=1 / 3, base=1500.0):
    """Walk through games in order; return (games with elo columns, final ratings).

    k:       how much a single game moves the ratings (bigger = reacts faster)
    hfa:     home-field advantage, in rating points
    regress: fraction each team is pulled back toward average each new season
    base:    the starting/average rating

    Adds: elo_home, elo_away (pre-game ratings), elo_prob (P home win).
    Games without a score (upcoming) get a prediction but no update.
    """
    # Work on a copy so we don't change the caller's data
    df = games.copy()
    df["home_team"] = df["home_team"].replace(TEAM_FIX)
    df["away_team"] = df["away_team"].replace(TEAM_FIX)

    # Games MUST be processed in time order, or we'd leak the future
    df = df.sort_values(["season", "week", "gameday"], kind="stable").reset_index(drop=True)

    has_loc = "location" in df.columns  # used to detect neutral-site games
    ratings, last_season = {}, None     # ratings: team -> current Elo
    elo_home, elo_away, probs = [], [], []

    for row in df.itertuples(index=False):
        # Start of a new season: rosters change, so shrink every rating
        # part of the way back toward the average
        if last_season is not None and row.season != last_season:
            ratings = {t: r + (base - r) * regress for t, r in ratings.items()}
        last_season = row.season

        # Look up each team's rating (new teams start at the base rating)
        h = ratings.get(row.home_team, base)
        a = ratings.get(row.away_team, base)
        neutral = has_loc and getattr(row, "location", "Home") == "Neutral"

        # PREDICT first, using only ratings from before this game
        p = expected_home_win(h, a, hfa, neutral)
        elo_home.append(h); elo_away.append(a); probs.append(p)

        # Upcoming game with no score yet: we keep the prediction, skip the update
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        # UPDATE after the result is known
        margin = row.home_score - row.away_score
        result = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)  # home win/tie/loss

        if margin == 0:
            mult = 1.0
        else:
            # Margin-of-victory multiplier: bigger wins move ratings more, but
            # with diminishing returns (log), and less if the winner was
            # already the heavy favorite (so blowouts by favorites don't inflate ratings)
            hfa_adj = 0.0 if neutral else hfa
            winner_diff = (h + hfa_adj - a) if margin > 0 else (a - h - hfa_adj)
            mult = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)

        # The surprise (result minus predicted probability) drives the change:
        # an upset moves ratings a lot, an expected win moves them a little
        delta = k * mult * (result - p)
        ratings[row.home_team] = h + delta
        ratings[row.away_team] = a - delta  # what one team gains, the other loses

    # Attach the pre-game ratings and predictions to the table
    df["elo_home"], df["elo_away"], df["elo_prob"] = elo_home, elo_away, probs
    return df, ratings


def vegas_prob(spread_line, sd=13.5):
    """Rough win prob from the point spread (nflverse: positive = home favored).

    Assumes the final margin is roughly bell-shaped around the spread with a
    standard deviation of ~13.5 points. An approximation, not the true market price.
    """
    return norm.cdf(np.asarray(spread_line, dtype=float) / sd)


def _metrics(y, p):
    """Score a set of predicted probabilities p against actual outcomes y (1 = home win)."""
    # Keep probabilities away from exactly 0 or 1 so log loss doesn't blow up
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "games": len(y),
        "accuracy": float(((p > 0.5) == y).mean()),  # how often the favored side won
        "log_loss": float(log_loss(y, p)),           # punishes confident wrong picks (lower = better)
        "brier": float(brier_score_loss(y, p)),      # avg squared probability error (lower = better)
    }


def evaluate(df, prob_col="elo_prob", start_season=2018):
    """Compare the model vs always-home vs Vegas on regular-season games >= start_season."""
    # Only completed games, and drop ties (they have no clear winner)
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played = played[played["home_score"] != played["away_score"]]
    played["home_win"] = (played["home_score"] > played["away_score"]).astype(int)

    # Seasons before start_season only warm up the ratings; we use them
    # to get the "always pick home" baseline rate
    train = played[played["season"] < start_season]
    home_rate = train["home_win"].mean() if len(train) else 0.57

    # Score only regular-season games from start_season onward
    test = played[(played["season"] >= start_season) & (played.get("game_type", "REG") == "REG")]
    y = test["home_win"].to_numpy()

    rows = {
        "Always home": _metrics(y, np.full(len(y), home_rate)),  # dumb baseline
        "Elo": _metrics(y, test[prob_col].to_numpy()),           # our model
    }
    # Add the Vegas benchmark if the spread column exists
    if "spread_line" in test.columns:
        t = test.dropna(subset=["spread_line"])
        rows["Vegas (spread-implied)"] = _metrics(t["home_win"].to_numpy(), vegas_prob(t["spread_line"]))
    return pd.DataFrame(rows).T