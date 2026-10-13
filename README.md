# NFL Weekly Game Predictor (2026 Season)

A machine learning pipeline that predicts NFL game winners with calibrated win
probabilities, plus an experimental against-the-spread (ATS) layer.

## Approach
1. **Baseline:** Elo ratings, benchmarked against "always pick home" and the Vegas line
2. **Features:** rolling 6-game EPA per play for offense and defense (nflverse data)
3. **Model:** logistic regression on Elo and EPA gaps
4. **Evaluation:** walk-forward testing by season (train on the past, predict the next
   season), so the model never sees future games

## Results (2018-2025 backtest, ~2,100 games)
| Model | Accuracy | Log loss |
|---|---|---|
| Always pick home | 54.0% | 0.691 |
| Elo | 63.4% | 0.637 |
| Logistic regression | 64.2% | 0.634 |
| Vegas line | 66.4% | 0.610 |

- The model is well calibrated: stated confidence is within about 2 points of the actual
  win rate in most bands (slightly overconfident above 75%).
- Rest days, split passing/rushing EPA, and gradient boosting did not improve results.
  Adding the model's features on top of the Vegas spread added nothing, so the market
  already prices in this information.
- ATS: the margin model picked the covering side 48.1% of the time (break-even is 52.4%),
  so it is tracked as an experiment, not betting advice.

## Project structure
- `explore.ipynb`: data loading, features, model comparison, weekly predictions
- `src/elo_baseline.py`: Elo model and evaluation helpers
- `app/streamlit_app.py`: dashboard for the latest week's predictions
- `data/predictions/`: weekly predictions, saved before kickoff

## Run the dashboard
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

## Next steps
Quarterback and injury data, a live scorecard, and automated weekly runs.
