# NBA Moneyline Prediction Model

A machine learning ensemble that identifies value betting opportunities in NBA moneyline markets. Trained on 5 seasons of historical data, validated against real closing odds from DraftKings/FanDuel/Caesars via SBR.

**Live dashboard:** https://nba-moneyline-model.streamlit.app

---

## Results (Clean Out-of-Sample)

| Season | Bets | Win Rate | ROI | Status |
|--------|------|----------|-----|--------|
| 2023-24 | 111 | 27.4% | +11.7% | Validation |
| 2024-25 | 148 | 28.4% | +3.3% | ✅ Clean test |
| 2025-26 | 130 | 30.0% | +7.4% | ✅ Clean test |

**Model:** AUC 0.729 | Accuracy 67.5% | 138 features

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv nba_model
nba_model\Scripts\activate        # Windows
source nba_model/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (first time)
python run_pipeline.py --fetch --features --train

# 4. Daily predictions
python predict.py

# 5. Launch dashboard
python -m streamlit run dashboard_app.py
```

---

## Project Structure

```
├── predict.py                  # Daily prediction script
├── dashboard_app.py            # Streamlit web dashboard
├── run_pipeline.py             # Master CLI (fetch/features/train/predict)
├── backtest_real_odds.py       # Backtest against real SBR closing lines
├── bankroll_sim.py             # Simulate bankroll growth from backtest CSV
├── analyze_clean_seasons.py    # Filter optimisation on clean seasons
├── requirements.txt
├── packages.txt                # System dependencies for Streamlit Cloud
├── config/
│   └── settings.py             # All parameters and betting filters
├── src/
│   ├── scraper.py              # NBA game log scraper (nba_api)
│   ├── features.py             # Rolling stats, Elo, eFG% engineering
│   ├── model.py                # Ensemble training and inference
│   ├── elo.py                  # Elo rating system
│   ├── edge.py                 # Vig removal, Kelly sizing, filters
│   ├── odds_scraper.py         # Live odds (SBR primary, Action Network fallback)
│   ├── sbr_scraper.py          # SBR historical and live odds scraper
│   └── predict.py              # Core prediction logic
├── models/                     # Trained model pkl files
├── data/
│   ├── processed/
│   │   └── game_features.parquet
│   ├── cache/                  # gitignored
│   └── odds/                   # gitignored
├── logs/
│   ├── backtest_real_2024.csv
│   ├── backtest_real_2025.csv
│   ├── backtest_real_2026.csv
│   ├── predictions_YYYY-MM-DD.csv
│   └── bet_tracker.json
└── .github/
    └── workflows/
        └── daily_predictions.yml
```

---

## Pipeline Commands

```bash
python run_pipeline.py --fetch              # Fetch latest game data
python run_pipeline.py --features           # Rebuild feature matrix
python run_pipeline.py --train              # Retrain all models
python predict.py                           # Daily predictions
python predict.py --season 2025-26          # Specific season
python predict.py --odds "LAC:-180,SAS:+155;DAL:+230,ORL:-280"  # Manual odds
```

---

## Backtesting

```bash
# Scrape historical odds (once per season)
python src/sbr_scraper.py --from 2024-10-22 --to 2025-04-13 --out data/odds/sbr_2024_25.csv --ml-only

# Backtest a season
python backtest_real_odds.py --csv data/odds/sbr_2024_25.csv --file_season 2025 --edge 15

# All seasons
python backtest_real_odds.py --all --edge 15

# Filter optimisation
python analyze_clean_seasons.py

# Bankroll simulation
python bankroll_sim.py --all --bankroll 10000 --stake-pct 2
```

---

## Model Architecture

**Ensemble:** LR 25% + XGBoost 35% + LightGBM 35% + Elo 5%

**Features (138):** Rolling stats, Elo ratings, eFG%, rest/schedule

**Split:** Train 2018-23 | Valid 2023-24 | Test 2024-25 + 2025-26

---

## Betting Filters (`config/settings.py`)

| Filter | Value | Reason |
|--------|-------|--------|
| Min edge | 15% | Below 15% ROI turns negative |
| Underdogs only | True | Favourites consistently negative ROI |
| Min odds | +150 | Near-even odds unreliable |
| Max odds | +500 | Longshots too volatile |
| Max edge cap | 30% | Model overconfident above 30% |
| Kelly fraction | 0.25 | Quarter Kelly |
| Max bet | 3% | Max 3% of bankroll per bet |

---

## Deployment

- **Dashboard:** Streamlit Cloud — https://nba-moneyline-model.streamlit.app
- **Automation:** GitHub Actions runs `predict.py` daily at 7am MYT (11pm UTC)
- **Season switch:** Auto-detects season from date — switches to 2026-27 in October 2026

**Season end checklist:**
1. Scrape full season odds
2. Run final backtest
3. Retrain: `python run_pipeline.py --features --train`
4. Push: `git add models/ data/processed/ && git push`

---

## License

Private. All rights reserved.
