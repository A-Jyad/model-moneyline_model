import sys
from pathlib import Path

# Ensure project root is on sys.path however the script is invoked
_SRC_DIR  = Path(__file__).resolve().parent          # .../nba_predictor/src
_ROOT_DIR = _SRC_DIR.parent                          # .../nba_predictor
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
"""
predict.py — Live prediction for today's NBA slate.

Usage:
  python predict.py
  python predict.py --date 2025-03-30
  python predict.py --odds  # prompts for manual odds input

Workflow:
  1. Loads today's schedule from NBA Stats API
  2. Computes current team rolling stats from cached game logs
  3. Runs ensemble model → P(home win)
  4. Compares to user-supplied or synthetic odds
  5. Outputs a formatted slate report + flagged bets
"""

import logging
import argparse
from datetime import datetime, date

import pandas as pd
import numpy as np


from config.settings import PROC_DIR, CACHE_DIR, MIN_EDGE_PCT, LOG_DIR
from src.scraper import fetch_schedule, fetch_injury_report, fetch_season_game_log
from src.features import clean_gamelogs, build_team_rolling_features, build_elo_ratings, get_feature_columns, get_injury_features_for_game
from src.odds_scraper import get_odds_dict
from src.edge import evaluate_game, implied_prob_from_american, log_bet
from src.elo import EloSystem

log = logging.getLogger("predict")


# ── Team state builder ────────────────────────────────────────────────────────

def get_current_team_states(season: str | None = None) -> pd.DataFrame:
    """
    Build current rolling feature state for all teams.
    Uses the most recent game per team from the cached game log.
    """
    if season is None:
        season = get_current_season()
    from src.scraper import fetch_season_game_log
    raw = fetch_season_game_log(season, "Regular Season")
    if raw.empty:
        log.error("No game log data available.")
        return pd.DataFrame()

    df = clean_gamelogs(raw)
    df["SEASON"] = season
    df = build_team_rolling_features(df)
    df = build_elo_ratings(df)

    # Get latest state per team (most recent game)
    latest = df.sort_values("GAME_DATE").groupby("TEAM_ABBREVIATION").last().reset_index()
    return latest


def build_prediction_row(home_state: pd.Series, away_state: pd.Series) -> dict:
    """
    Build a single game feature row from two team state rows.
    Mirrors the structure of game_features.parquet.
    """
    from config.settings import BOX_COLS, ROLLING_WINDOWS

    row = {}

    # Rolling features
    for col in BOX_COLS:
        for suffix in [f"_roll{w}" for w in ROLLING_WINDOWS] + ["_ewm"]:
            h_col = f"{col}{suffix}"
            row[f"HOME_{h_col}"] = home_state.get(h_col, np.nan)
            row[f"AWAY_{h_col}"] = away_state.get(h_col, np.nan)
            row[f"DIFF_{h_col}"] = row[f"HOME_{h_col}"] - row[f"AWAY_{h_col}"]

    # Elo
    row["HOME_ELO_PRE"] = home_state.get("ELO_PRE", 1500)
    row["AWAY_ELO_PRE"] = away_state.get("ELO_PRE", 1500)
    row["ELO_DIFF"]     = row["HOME_ELO_PRE"] - row["AWAY_ELO_PRE"]

    # Injury features (default 0 — caller injects live values for live predictions)
    row["home_players_out"]   = 0
    row["away_players_out"]   = 0
    row["home_injury_impact"] = 0.0
    row["away_injury_impact"] = 0.0
    row["injury_impact_diff"] = 0.0

    # Rest
    row["HOME_days_rest"] = home_state.get("days_rest", 2)
    row["AWAY_days_rest"] = away_state.get("days_rest", 2)
    row["REST_DIFF"]      = row["HOME_days_rest"] - row["AWAY_days_rest"]
    row["HOME_IS_B2B"]    = home_state.get("is_b2b", 0)
    row["AWAY_IS_B2B"]    = away_state.get("is_b2b", 0)

    # Streaks
    row["HOME_win_streak"] = home_state.get("win_streak", 0)
    row["AWAY_win_streak"] = away_state.get("win_streak", 0)
    row["STREAK_DIFF"]     = row["HOME_win_streak"] - row["AWAY_win_streak"]
    row["AWAY_away_streak"]= away_state.get("away_streak", 0)

    return row


# ── Main Predict Function ─────────────────────────────────────────────────────

def get_current_season(target_date: str | None = None) -> str:
    """
    Auto-detect the NBA season string from a date.
    NBA seasons run Oct-Jun. e.g. Oct 2025 - Jun 2026 = "2025-26"
    """
    from datetime import date as date_cls
    d = date_cls.fromisoformat(target_date) if target_date else date_cls.today()
    year = d.year
    # If month is Oct-Dec, season starts this year
    if d.month >= 10:
        return f"{year}-{str(year+1)[-2:]}"
    # Jan-Sep, season started last year
    else:
        return f"{year-1}-{str(year)[-2:]}"


def predict_today(
    target_date: str | None = None,
    odds_dict: dict | None = None,   # manual odds: {(home,away): {"home":ml,"away":ml}}
    season: str | None = None,
    min_edge: float | None = None,
) -> pd.DataFrame:
    """
    Generate predictions for all games on target_date.

    odds_dict: {game_id: {"home": american_home, "away": american_away}}
               If None, outputs model probabilities only (no edge calc).
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    # Auto-detect season if not provided
    if season is None:
        season = get_current_season(target_date)

    # Use default edge from settings if not specified
    if min_edge is None:
        min_edge = MIN_EDGE_PCT

    log.info(f"Generating predictions for {target_date} | season: {season}")

    # Load schedule
    schedule = fetch_schedule(season)
    if schedule.empty:
        log.error("Could not load schedule.")
        return pd.DataFrame()

    schedule["game_date"] = pd.to_datetime(schedule["game_date"]).dt.strftime("%Y-%m-%d")
    today_games = schedule[schedule["game_date"] == target_date].copy()

    if today_games.empty:
        log.info(f"No games scheduled for {target_date}.")
        return pd.DataFrame()

    log.info(f"Found {len(today_games)} games for {target_date}")

    # Load current team states
    team_states = get_current_team_states(season)
    if team_states.empty:
        log.error("No team states available.")
        return pd.DataFrame()

    state_by_team = {row["TEAM_ABBREVIATION"]: row for _, row in team_states.iterrows()}

    # Load models
    try:
        from src.model import NBAEnsemble
        model = NBAEnsemble().load()
        feat_cols = model.feat_cols
        log.info(f"Model loaded. Expects {len(feat_cols)} features.")
    except FileNotFoundError:
        log.warning("Models not trained yet — using Elo only.")
        model = None
        feat_cols = []

    # Load Elo
    elo = EloSystem()
    from config.settings import CACHE_DIR
    if (CACHE_DIR / "elo_state.json").exists():
        elo.load()
    else:
        # Cloud: recompute Elo from committed game_features.parquet
        log.info("No Elo cache — recomputing from game_features.parquet...")
        try:
            from config.settings import PROC_DIR
            gf = pd.read_parquet(PROC_DIR / "game_features.parquet")
            elo.fit(gf)
            log.info(f"Elo recomputed from {len(gf)} games: {len(elo.ratings)} teams")
        except Exception as e:
            log.warning(f"Elo recompute failed: {e} — ratings will be default 1500")

    # Load injuries
    injuries = fetch_injury_report()

    # Auto-fetch live odds (uses Odds API key if set, else Action Network)
    log.info("Fetching live odds...")
    try:
        # Clear stale cache before fetching
        from config.settings import CACHE_DIR
        stale = CACHE_DIR / "odds_live.json"
        if stale.exists():
            stale.unlink()
        live_odds = get_odds_dict(force_refresh=True)
        if live_odds:
            log.info(f"Live odds loaded: {len(live_odds)} matchups")
        else:
            log.info("No live odds available — model probabilities only")
    except Exception as e:
        log.warning(f"Odds fetch failed: {e}")
        live_odds = {}

    # Merge manual odds on top (manual takes priority over auto-fetched)
    if odds_dict:
        # Normalise keys to uppercase
        manual = {(h.upper(), a.upper()): v for (h, a), v in odds_dict.items()}
        live_odds.update(manual)
        log.info(f"Manual odds merged: {len(manual)} games")

    results = []

    for _, game in today_games.iterrows():
        home = str(game.get("home_team", ""))
        away = str(game.get("away_team", ""))
        gid  = str(game.get("game_id", ""))

        if home not in state_by_team or away not in state_by_team:
            log.warning(f"  Missing state for {home} vs {away} — skipping.")
            continue

        home_state = state_by_team[home]
        away_state = state_by_team[away]

        feat_row = build_prediction_row(home_state, away_state)

        # Elo probability
        elo_prob = elo.win_probability(home, away)

        # Ensemble probability
        if model is not None:
            # Build feature vector — only use cols that exist in feat_cols
            x_vals = [float(feat_row.get(c, 0) or 0) for c in feat_cols]
            X       = np.array([x_vals])
            elo_arr = np.array([elo_prob])
            comps   = model.predict_proba_components(X, elo_arr)
            # blend() returns an array — take the first (and only) element
            blended = model.blend(comps)
            p_home  = float(np.asarray(blended).flat[0])
        else:
            p_home = elo_prob

        # Injury flags
        home_out = injuries[injuries["team"].str.upper().str.contains(home, na=False) &
                            injuries["status"].isin(["Out", "Doubtful"])]["player"].tolist()
        away_out = injuries[injuries["team"].str.upper().str.contains(away, na=False) &
                            injuries["status"].isin(["Out", "Doubtful"])]["player"].tolist()

        # Edge evaluation — use live odds if available, else passed odds_dict
        game_odds = None
        if odds_dict and gid in odds_dict:
            game_odds = odds_dict[gid]
        elif live_odds.get((home, away)):
            game_odds = live_odds[(home, away)]

        if game_odds:
            ev = evaluate_game(
                home_team=home, away_team=away,
                model_prob_home=p_home,
                home_american_odds=game_odds["home"],
                away_american_odds=game_odds["away"],
                game_date=target_date, game_id=gid,
                min_edge=min_edge,
            )
        else:
            ev = {
                "has_edge": False,
                "recommendation": "No odds available — model only",
                "edge_home_pct": None,
                "edge_away_pct": None,
                "kelly_units": 0,
            }

        # Get structured injury features
        inj_feats = get_injury_features_for_game(home, away, injuries)

        # Extract odds from ev or live_odds dict
        game_odds = live_odds.get((home, away), live_odds.get((home.upper(), away.upper()), {}))
        home_ml_val = game_odds.get("home") if game_odds else None
        away_ml_val = game_odds.get("away") if game_odds else None

        results.append({
            "game_id":              gid,
            "date":                 target_date,
            "home_team":            home,
            "away_team":            away,
            "p_home_win":           round(p_home, 4),
            "p_away_win":           round(1 - p_home, 4),
            "elo_home":             round(elo.get_rating(home), 1),
            "elo_away":             round(elo.get_rating(away), 1),
            "elo_diff":             round(elo.get_rating(home) - elo.get_rating(away), 1),
            "rest_home":            int(home_state.get("days_rest", 2)),
            "rest_away":            int(away_state.get("days_rest", 2)),
            "home_b2b":             bool(home_state.get("is_b2b", 0)),
            "away_b2b":             bool(away_state.get("is_b2b", 0)),
            "home_streak":          int(home_state.get("win_streak", 0)),
            "away_streak":          int(away_state.get("win_streak", 0)),
            "home_injuries":        ", ".join(home_out) if home_out else "None",
            "away_injuries":        ", ".join(away_out) if away_out else "None",
            "home_players_out":     inj_feats["home_players_out"],
            "away_players_out":     inj_feats["away_players_out"],
            "home_injury_impact":   inj_feats["home_injury_impact"],
            "away_injury_impact":   inj_feats["away_injury_impact"],
            "injury_impact_diff":   inj_feats["injury_impact_diff"],
            "home_ml":              home_ml_val,
            "away_ml":              away_ml_val,
            "edge_home_pct":        ev.get("edge_home_pct"),
            "edge_away_pct":        ev.get("edge_away_pct"),
            "has_edge":             ev.get("has_edge", False),
            "kelly_units":          ev.get("kelly_units", 0),
            "recommendation":       ev.get("recommendation", ""),
        })

    df = pd.DataFrame(results)

    # Save predictions
    out_path = LOG_DIR / f"predictions_{target_date}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Predictions saved: {out_path}")

    return df


def print_slate_report(df: pd.DataFrame):
    """Print a formatted slate report to stdout."""
    print("\n" + "="*70)
    print(f"  NBA PREDICTION SLATE — {df['date'].iloc[0] if len(df) else 'N/A'}")
    print("="*70)

    if df.empty:
        print("  No games found.")
        return

    for _, row in df.iterrows():
        prob_bar = "█" * int(row["p_home_win"] * 20) + "░" * (20 - int(row["p_home_win"] * 20))
        b2b_h = " [B2B]" if row["home_b2b"] else ""
        b2b_a = " [B2B]" if row["away_b2b"] else ""

        print(f"\n  {row['home_team']}{b2b_h} vs {row['away_team']}{b2b_a}")
        print(f"  [{prob_bar}] {row['p_home_win']*100:.1f}% / {row['p_away_win']*100:.1f}%")
        print(f"  Elo: {row['elo_home']} vs {row['elo_away']} (diff: {row['elo_diff']:+.0f})")

        if row["home_injuries"] != "None":
            print(f"  ⚠ {row['home_team']} OUT: {row['home_injuries']}")
        if row["away_injuries"] != "None":
            print(f"  ⚠ {row['away_team']} OUT: {row['away_injuries']}")

        if row["has_edge"]:
            print(f"  ★ {row['recommendation']}")
        else:
            print(f"  ✗ {row['recommendation']}")

    # Flagged bets summary
    flagged = df[df["has_edge"]]
    print(f"\n{'='*70}")
    print(f"  FLAGGED BETS: {len(flagged)} / {len(df)}")
    if len(flagged) > 0:
        for _, row in flagged.iterrows():
            print(f"  → {row['recommendation']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--season", default=None, help="Season string (default: auto-detect)")
    parser.add_argument("--edge", type=float, default=MIN_EDGE_PCT)
    args = parser.parse_args()

    preds = predict_today(
        target_date=args.date,
        season=args.season,
        min_edge=args.edge,
    )
    print_slate_report(preds)