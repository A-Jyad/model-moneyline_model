import sys
from pathlib import Path

# Ensure project root is on sys.path however the script is invoked
_SRC_DIR  = Path(__file__).resolve().parent          # .../nba_predictor/src
_ROOT_DIR = _SRC_DIR.parent                          # .../nba_predictor
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
"""
edge.py — Edge detection and Kelly criterion bet sizing.

Core logic:
  1. Convert American odds → decimal odds → implied probability
  2. Remove vig to get the "fair" market probability
  3. Compare model probability to fair market prob
  4. Flag bets where edge ≥ MIN_EDGE_PCT
  5. Size bets using fractional Kelly criterion
"""

import logging
import csv
from datetime import datetime

import pandas as pd
import numpy as np


from config.settings import (
    MIN_EDGE_PCT, KELLY_FRACTION, MAX_BET_PCT, MIN_BET_UNITS,
    VIG_REMOVE_METHOD, BET_LOG_FILE, BET_LOG_COLS,
    BET_UNDERDOGS_ONLY, BET_MAX_ODDS, BET_MIN_ODDS, BET_MAX_EDGE,
)

log = logging.getLogger("edge")


# ── Odds Conversion ───────────────────────────────────────────────────────────

def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return american / 100 + 1
    else:
        return 100 / abs(american) + 1


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds back to American."""
    if decimal >= 2.0:
        return (decimal - 1) * 100
    else:
        return -100 / (decimal - 1)


def decimal_to_implied_prob(decimal: float) -> float:
    """Convert decimal odds to raw implied probability (includes vig)."""
    return 1 / decimal


def remove_vig(prob_home: float, prob_away: float,
               method: str = VIG_REMOVE_METHOD) -> tuple[float, float]:
    """
    Remove sportsbook vig from a two-way market.

    Multiplicative (default): divide each prob by the overround.
    Additive: subtract half the vig from each side.
    """
    overround = prob_home + prob_away

    if method == "multiplicative":
        fair_home = prob_home / overround
        fair_away = prob_away / overround
    elif method == "additive":
        vig = overround - 1
        fair_home = prob_home - vig / 2
        fair_away = prob_away - vig / 2
    else:
        raise ValueError(f"Unknown vig removal method: {method}")

    return fair_home, fair_away


def implied_prob_from_american(home_odds: float, away_odds: float) -> tuple[float, float]:
    """
    Convert a two-sided American odds line to vig-free implied probabilities.
    Returns (fair_prob_home, fair_prob_away).
    """
    raw_home = decimal_to_implied_prob(american_to_decimal(home_odds))
    raw_away = decimal_to_implied_prob(american_to_decimal(away_odds))
    return remove_vig(raw_home, raw_away)


# ── Edge Calculation ──────────────────────────────────────────────────────────

def calculate_edge(model_prob: float, market_prob: float) -> float:
    """Edge = model probability minus fair market probability, in percentage points."""
    return (model_prob - market_prob) * 100


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """
    Expected value per unit staked.
    EV = (model_prob × (decimal_odds − 1)) − (1 − model_prob)
    """
    return model_prob * (decimal_odds - 1) - (1 - model_prob)


# ── Kelly Criterion ───────────────────────────────────────────────────────────

def kelly_fraction_bet(model_prob: float, decimal_odds: float,
                        fraction: float = KELLY_FRACTION) -> float:
    """
    Compute fractional Kelly bet size as a fraction of bankroll.

    Full Kelly: f = (b*p - q) / b
      where b = decimal_odds - 1 (net odds)
            p = model win probability
            q = 1 - p (model loss probability)

    Returns 0 if Kelly is negative (no edge) or odds are invalid.
    """
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p

    if b <= 0:
        return 0.0

    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return 0.0

    return min(full_kelly * fraction, MAX_BET_PCT / 100)


def kelly_units(model_prob: float, decimal_odds: float,
                 bankroll_units: float = 100.0) -> float:
    """Return Kelly bet size in units given a bankroll."""
    frac = kelly_fraction_bet(model_prob, decimal_odds)
    units = frac * bankroll_units
    return max(round(units, 2), MIN_BET_UNITS if frac > 0 else 0)


# ── Bet Evaluation ────────────────────────────────────────────────────────────

def evaluate_game(
    home_team: str,
    away_team: str,
    model_prob_home: float,
    home_american_odds: float,
    away_american_odds: float,
    game_date: str = "",
    game_id: str = "",
    min_edge: float = MIN_EDGE_PCT,
    underdogs_only: bool = BET_UNDERDOGS_ONLY,
    max_odds: int = BET_MAX_ODDS,
    min_odds: int = BET_MIN_ODDS,
    max_edge: float = BET_MAX_EDGE,
) -> dict:
    """
    Full evaluation of a single game for betting value.

    Returns a dict with:
      - model vs market probabilities
      - edge for each side
      - recommended bet (if any)
      - Kelly sizing
      - EV
    """
    model_prob_away = 1 - model_prob_home

    fair_home, fair_away = implied_prob_from_american(home_american_odds, away_american_odds)

    edge_home = calculate_edge(model_prob_home, fair_home)
    edge_away = calculate_edge(model_prob_away, fair_away)

    home_dec = american_to_decimal(home_american_odds)
    away_dec = american_to_decimal(away_american_odds)

    ev_home = expected_value(model_prob_home, home_dec)
    ev_away = expected_value(model_prob_away, away_dec)

    # Determine best bet
    bet_side  = None
    bet_edge  = None
    bet_odds  = None
    bet_prob  = None
    bet_units = 0.0
    bet_ev    = 0.0

    if edge_home >= min_edge and edge_home >= edge_away:
        bet_side  = home_team
        bet_edge  = edge_home
        bet_odds  = home_american_odds
        bet_prob  = model_prob_home
        bet_units = kelly_units(model_prob_home, home_dec)
        bet_ev    = ev_home
    elif edge_away >= min_edge:
        bet_side  = away_team
        bet_edge  = edge_away
        bet_odds  = away_american_odds
        bet_prob  = model_prob_away
        bet_units = kelly_units(model_prob_away, away_dec)
        bet_ev    = ev_away



    # ── Apply betting filters ────────────────────────────────────────────────
    if bet_side is not None and bet_odds is not None:
        discard = False
        # 1. Underdogs only — favorites ROI: -9.2%
        if underdogs_only and bet_odds < 0:
            discard = True
        # 2. No extreme longshots — >+500 ROI: -12.5%
        if bet_odds > max_odds:
            discard = True
        # 3. No near-even odds — dead zone is approx -140 to +140, ROI: -27.0%
        # Keep only: real underdogs above +140 (already ensured by underdogs_only)
        # This cuts the near-even range where model has no reliable edge
        if bet_odds <= abs(min_odds):   # skip if odds <= +140
            discard = True
        # 4. No extreme edge — >30% edge means model overconfident, ROI: -33.5%
        if bet_edge is not None and bet_edge > max_edge:
            discard = True
        if discard:
            bet_side = bet_edge = bet_odds = bet_prob = None
            bet_units = bet_ev = 0.0

    return {
        "game_id":          game_id,
        "game_date":        game_date,
        "home_team":        home_team,
        "away_team":        away_team,
        "model_prob_home":  round(model_prob_home, 4),
        "model_prob_away":  round(model_prob_away, 4),
        "fair_prob_home":   round(fair_home, 4),
        "fair_prob_away":   round(fair_away, 4),
        "edge_home_pct":    round(edge_home, 2),
        "edge_away_pct":    round(edge_away, 2),
        "ev_home":          round(ev_home, 4),
        "ev_away":          round(ev_away, 4),
        "home_odds":        home_american_odds,
        "away_odds":        away_american_odds,
        "has_edge":         bet_side is not None,
        "bet_side":         bet_side,
        "bet_edge_pct":     round(bet_edge, 2) if bet_edge else None,
        "bet_odds":         bet_odds,
        "bet_prob":         round(bet_prob, 4) if bet_prob else None,
        "kelly_units":      bet_units,
        "expected_value":   round(bet_ev, 4),
        "recommendation":   _format_recommendation(bet_side, bet_edge, bet_odds, bet_units, bet_ev)
    }


def _format_recommendation(bet_side, edge, odds, units, ev) -> str:
    if bet_side is None:
        return "NO BET — insufficient edge"
    sign = "+" if odds > 0 else ""
    return (
        f"BET {bet_side} ({sign}{odds:.0f}) | "
        f"Edge: {edge:.1f}% | "
        f"Units: {units:.2f} | "
        f"EV: {ev:+.3f}"
    )


# ── Batch Evaluation ──────────────────────────────────────────────────────────

def evaluate_slate(predictions_df: pd.DataFrame,
                   odds_dict: dict | None = None,
                   min_edge: float = MIN_EDGE_PCT) -> pd.DataFrame:
    """
    Evaluate a full day's slate of games.

    predictions_df must have: GAME_ID, GAME_DATE, HOME_TEAM_ABBREVIATION,
                               AWAY_TEAM_ABBREVIATION, P_HOME_WIN
    odds_dict: {game_id: {"home": american_home, "away": american_away}}
               If None, uses default −110/−110 (no-vig market) for demo.

    Returns DataFrame of evaluated bets.
    """
    results = []
    for _, row in predictions_df.iterrows():
        gid = str(row.get("GAME_ID", ""))

        # Get odds
        if odds_dict and gid in odds_dict:
            home_odds = odds_dict[gid]["home"]
            away_odds = odds_dict[gid]["away"]
        else:
            # Default: treat as even market for backtest purposes
            home_odds = -110
            away_odds = -110

        result = evaluate_game(
            home_team          = str(row.get("HOME_TEAM_ABBREVIATION", "")),
            away_team          = str(row.get("AWAY_TEAM_ABBREVIATION", "")),
            model_prob_home    = float(row.get("P_HOME_WIN", 0.5)),
            home_american_odds = home_odds,
            away_american_odds = away_odds,
            game_date          = str(row.get("GAME_DATE", "")),
            game_id            = gid,
            min_edge           = min_edge,
        )
        results.append(result)

    return pd.DataFrame(results)


# ── Bet Logging ───────────────────────────────────────────────────────────────

def log_bet(bet_dict: dict, result: int | None = None, pnl: float | None = None):
    """
    Append a bet to the persistent bet log CSV.
    result: 1 if won, 0 if lost, None if pending
    """
    row = {
        "date":              bet_dict.get("game_date", datetime.now().strftime("%Y-%m-%d")),
        "game_id":           bet_dict.get("game_id", ""),
        "home_team":         bet_dict.get("home_team", ""),
        "away_team":         bet_dict.get("away_team", ""),
        "model_prob_home":   bet_dict.get("model_prob_home", ""),
        "market_implied_home": bet_dict.get("fair_prob_home", ""),
        "edge_pct":          bet_dict.get("bet_edge_pct", ""),
        "bet_side":          bet_dict.get("bet_side", ""),
        "american_odds":     bet_dict.get("bet_odds", ""),
        "kelly_units":       bet_dict.get("kelly_units", ""),
        "result":            result if result is not None else "PENDING",
        "pnl_units":         pnl if pnl is not None else "",
        "notes":             "",
    }

    write_header = not BET_LOG_FILE.exists()
    with open(BET_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BET_LOG_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_bet_log() -> pd.DataFrame:
    """Load the bet log CSV."""
    if BET_LOG_FILE.exists():
        return pd.read_csv(BET_LOG_FILE)
    return pd.DataFrame(columns=BET_LOG_COLS)


def bet_log_summary() -> dict:
    """Compute ROI and other summary stats from the bet log."""
    df = load_bet_log()
    df = df[df["result"] != "PENDING"].copy()
    if df.empty:
        return {"total_bets": 0, "roi": 0.0, "win_rate": 0.0}

    df["pnl_units"] = pd.to_numeric(df["pnl_units"], errors="coerce")
    df["kelly_units"] = pd.to_numeric(df["kelly_units"], errors="coerce")

    total_staked = df["kelly_units"].sum()
    total_pnl    = df["pnl_units"].sum()
    won          = (df["result"] == "1").sum()

    return {
        "total_bets":   len(df),
        "wins":         int(won),
        "losses":       int(len(df) - won),
        "win_rate":     round(won / len(df), 4) if len(df) > 0 else 0,
        "total_staked": round(total_staked, 2),
        "total_pnl":    round(total_pnl, 2),
        "roi_pct":      round(total_pnl / total_staked * 100, 2) if total_staked > 0 else 0,
    }


if __name__ == "__main__":
    # Quick demo
    result = evaluate_game(
        home_team="OKC", away_team="NYK",
        model_prob_home=0.68,
        home_american_odds=-150,
        away_american_odds=+130,
    )
    for k, v in result.items():
        print(f"  {k:25s}: {v}")