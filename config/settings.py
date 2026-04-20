"""
Central configuration for the NBA Prediction Model.
Edit this file to tune all parameters.
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = DATA_DIR / "cache"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
MODEL_DIR  = BASE_DIR / "models"
LOG_DIR    = BASE_DIR / "logs"

for d in [CACHE_DIR, RAW_DIR, PROC_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Scraper ──────────────────────────────────────────────────────────────────
# NBA Stats API — no key required, just polite headers + rate limiting
NBA_STATS_BASE  = "https://stats.nba.com/stats"
SCRAPE_HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer":        "https://www.nba.com/",
    "Accept":         "application/json, text/plain, */*",
    "Accept-Language":"en-US,en;q=0.9",
    "Origin":         "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
}
REQUEST_DELAY   = 1.2   # seconds between requests (be polite)
REQUEST_TIMEOUT = 30    # seconds

# Seasons to collect — format "YYYY-YY"
SEASONS = [
    "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25",
    "2025-26",
]
# Current active season (auto-used for schedule + injury fetching)
# Auto-detect current season based on today's date (switches in October)
def _get_current_season() -> str:
    from datetime import date
    d = date.today()
    if d.month >= 10:
        return f"{d.year}-{str(d.year+1)[-2:]}"
    return f"{d.year-1}-{str(d.year)[-2:]}"

CURRENT_SEASON = _get_current_season()
SEASON_TYPES = ["Regular Season"]   # add "Playoffs" if desired

# Cache filenames
CACHE_GAMELOG_INDEX = CACHE_DIR / "gamelog_index.json"   # tracks last fetched game per season
CACHE_ADVANCED_INDEX= CACHE_DIR / "advanced_index.json"

# ── Feature Engineering ──────────────────────────────────────────────────────
ROLLING_WINDOWS   = [5, 10, 20]      # games for rolling averages
DECAY_HALFLIFE    = 10               # exponential decay half-life (games)
HOME_COURT_EDGE   = 2.5              # points, used for Elo adjustment

# Core box-score columns to roll
BOX_COLS = [
    "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
    "REB", "AST", "TOV", "STL", "BLK",
    "PLUS_MINUS",
]

# ── Model ────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
TEST_SEASON      = "2024-25"         # held-out for backtesting
VALID_SEASON     = "2023-24"         # validation during training
# Additional test season — also fully held out (never seen in training)
TEST_SEASON_2    = "2025-26"

# Logistic Regression
LR_PARAMS = {"C": 0.1, "max_iter": 1000, "random_state": RANDOM_SEED}

# XGBoost
XGB_PARAMS = {
    "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "use_label_encoder": False, "eval_metric": "logloss",
    "random_state": RANDOM_SEED, "n_jobs": -1,
}

# LightGBM
LGB_PARAMS = {
    "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1,
}

# Ensemble weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {"lr": 0.25, "xgb": 0.35, "lgb": 0.35, "elo": 0.05}

# ── Elo ──────────────────────────────────────────────────────────────────────
ELO_K            = 20       # K-factor
ELO_START        = 1500     # starting rating
ELO_REGRESS_FRAC = 0.33     # regression to mean each new season

# ── Edge Detection ───────────────────────────────────────────────────────────
MIN_EDGE_PCT     = 15.0     # minimum edge % to flag a bet
                             # Below 12%: ROI -7.5% (not reliable)
                             # 12-15%:    ROI +25.4% (sweet spot)
VIG_REMOVE_METHOD= "multiplicative"  # "additive" or "multiplicative"

# ── Betting Filters ──────────────────────────────────────────────────────────
# Optimised on 2024-25 + 2025-26 (both fully out-of-sample)
# Best config: edge >= 15%, underdogs only, odds +141 to +500
# Result: +5.3% ROI, ~147 bets/season, both seasons positive (+6%, +5%)
BET_UNDERDOGS_ONLY = True    # favorites consistently negative across all seasons
BET_MAX_ODDS       = 1000     # >+500 adds noise; sweet spot is +141 to +500
BET_MIN_ODDS       = -140    # skip near-even odds (+100 to +140 dead zone)
BET_MAX_EDGE       = 30.0    # >30% = model overconfident, ROI deteriorates


# ── Bet Sizing ───────────────────────────────────────────────────────────────
KELLY_FRACTION   = 0.25     # fractional Kelly (0.25 = quarter Kelly)
MAX_BET_PCT      = 3.0      # never bet more than 3% of bankroll on one game
MIN_BET_UNITS    = 0.5      # minimum bet in units

# ── Logging ──────────────────────────────────────────────────────────────────
BET_LOG_FILE     = LOG_DIR / "bet_log.csv"
BET_LOG_COLS     = [
    "date", "game_id", "home_team", "away_team",
    "model_prob_home", "market_implied_home",
    "edge_pct", "bet_side", "american_odds",
    "kelly_units", "result", "pnl_units", "notes",
]