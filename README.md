# NFL Weekly Game Predictor (2026 Season)

A machine learning pipeline that predicts NFL game winners with calibrated win
probabilities, plus an experimental against-the-spread (ATS) layer.

**[Live dashboard](https://nfl-weekly-predictor.streamlit.app/)**

## Approach
1. **Baseline:** Elo ratings, benchmarked against "always pick home" and the Vegas line
2. **Features:** rolling 6-game EPA per play for offense and defense (nflverse data)
3. **Model:** logistic regression on Elo and EPA gaps
4. **Evaluation:** walk-forward testing by season (train on the past, predict the next
   season), so the model never sees future games
5. **Automation:** a GitHub Actions job runs every Tuesday. It predicts the next week,
   scores past predictions against real results, and updates the dashboard

## Tech stack
| Layer | Tools |
|---|---|
| Language | Python |
| Data source | nflverse (`nflreadpy`) |
| Data processing | pandas, NumPy |
| Modeling | scikit-learn (logistic regression, Ridge, gradient boosting), SciPy |
| Dashboard | Streamlit |
| Automation | GitHub Actions |
| Development | VS Code, Jupyter notebooks |
| Version control | Git, GitHub |

## Results (2018-2025 backtest, ~2,100 games)
| Model | Accuracy | Log loss |
|---|---|---|
| Always pick home | 54.0% | 0.691 |
| Elo (tuned) | 63.9% | 0.636 |
| Logistic regression | 64.1% | 0.634 |
| Vegas line | 66.4% | 0.610 |

- The model is well calibrated: stated confidence is within about 2 points of the actual
  win rate in most bands (slightly overconfident above 75%).
- Elo settings were tuned on 2018-2022 and checked on 2023-2025. A smaller home-field
  advantage fit recent seasons better, but the gain was tiny (about 0.001 log loss).
- Rest days, split passing/rushing EPA, and gradient boosting did not improve results.
  Adding the model's features on top of the Vegas spread added nothing, so the market
  already prices in this information.
- ATS: the margin model picked the covering side about 48% of the time (break-even is
  52.4%), so it is tracked as an experiment, not betting advice.

## Project structure
- `explore.ipynb`: data loading, features, model comparison, calibration, ATS
- `src/elo_baseline.py`: Elo model and evaluation helpers
- `src/weekly.py`: trains the models and saves the next week's predictions
- `src/score.py`: scores saved predictions against real results
- `app/streamlit_app.py`: dashboard with the latest picks and a scorecard
- `.github/workflows/weekly.yml`: Tuesday automation
- `data/predictions/`: weekly predictions, saved before kickoff
- `data/scorecard.csv`: game-by-game results of past predictions

## Run the dashboard
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

## Limitations
- **No quarterback, injury, or weather data.** The model only sees team-level results, so
  it can't react to a starter being out. This is the most likely reason it trails Vegas.
- **Does not beat the Vegas line.** The backtest accuracy is about 2 points below the
  market, and adding the model's features on top of the spread added nothing.
- **ATS layer shows no edge.** It picked the covering side about 48% of the time, below
  the 52.4% break-even, so it is a tracked experiment and not betting advice.
- **Small sample.** There are only ~270 games a season, which limits how much a model can
  learn and makes small differences between models (under about 1 point) mostly noise.
- **Early-season predictions are weaker.** Rolling form is based on few games in the first
  weeks, so it leans more on Elo carried over from last season.
- **Overconfident at the top.** Picks shown at 75%+ won about 76% of the time in the
  backtest, versus 81% claimed on average.
- **Approximate Vegas benchmark.** Vegas win probability is converted from the point
  spread with a normal approximation, not taken from actual moneyline prices.
- **Lightly tuned.** Only the Elo home-field value was tuned; K and season carryover were
  checked at a few values, and the 6-game rolling window was not optimized.
- **Line timing.** Live predictions use the spread available when the Tuesday job runs,
  which may differ from the final lines used in the backtest.

## Next steps
Quarterback and injury data, and tracking live results through the 2026 season.