import sys
from pathlib import Path

# Ensure project root is on sys.path however the script is invoked
_SRC_DIR  = Path(__file__).resolve().parent          # .../nba_predictor/src
_ROOT_DIR = _SRC_DIR.parent                          # .../nba_predictor
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
"""
backtest.py — Historical simulation engine.

Simulates betting on the test season using:
  - Model predictions (from ensemble)
  - Synthetic market odds based on real implied probabilities
  - Kelly sizing with bankroll tracking
  - Full P&L accounting

Outputs:
  - Detailed bet-by-bet CSV
  - Summary stats: ROI, Sharpe, win rate, CLV proxy
  - Calibration curve data
"""

import logging

import numpy as np
import pandas as pd


from config.settings import (
    TEST_SEASON, PROC_DIR, LOG_DIR, MIN_EDGE_PCT, KELLY_FRACTION,
)
from src.edge import (
    evaluate_slate, american_to_decimal, kelly_units,
    implied_prob_from_american,
)

log = logging.getLogger("backtest")


# ── Synthetic Odds ────────────────────────────────────────────────────────────

def elo_prob_to_american(p_home: float, vig: float = 0.045) -> tuple[float, float]:
    """
    Convert a true win probability to a two-sided American line with vig.
    This simulates what a sharp book would offer.
    """
    p_away = 1 - p_home

    # Apply vig
    p_home_v = p_home * (1 + vig)
    p_away_v = p_away * (1 + vig)

    def to_american(p):
        if p >= 0.5:
            return -(p / (1 - p)) * 100
        else:
            return ((1 - p) / p) * 100

    return round(to_american(p_home_v)), round(to_american(p_away_v))


def generate_synthetic_odds(df: pd.DataFrame) -> dict:
    """
    Generate synthetic market odds for each game using Elo probabilities.
    In real deployment, replace with actual scraped odds.

    Returns {game_id: {"home": american, "away": american}}
    """
    odds_dict = {}
    for _, row in df.iterrows():
        gid = str(row.get("GAME_ID", ""))
        # Use Elo differential as "market" basis
        elo_diff = row.get("ELO_DIFF", 0)
        p_home_market = 1 / (1 + 10 ** (-elo_diff / 400))

        home_odds, away_odds = elo_prob_to_american(p_home_market)
        odds_dict[gid] = {"home": home_odds, "away": away_odds}
    return odds_dict


# ── Bankroll Simulation ───────────────────────────────────────────────────────

def simulate_bankroll(bets_df: pd.DataFrame, starting_bankroll: float = 100.0) -> pd.DataFrame:
    """
    Simulate bankroll evolution over the test season.
    bets_df must have: kelly_units, has_edge, result_correct (1/0), bet_odds
    """
    df = bets_df[bets_df["has_edge"]].copy().reset_index(drop=True)
    if df.empty:
        return df

    bankroll = starting_bankroll
    bankroll_history = []
    stake_history    = []
    pnl_history      = []

    for _, row in df.iterrows():
        # Kelly units as fraction of current bankroll
        from src.edge import kelly_fraction_bet
        stake_frac = kelly_fraction_bet(
            float(row["bet_prob"]),
            american_to_decimal(float(row["bet_odds"]))
        )
        stake = stake_frac * bankroll

        if row["result_correct"] == 1:
            dec_odds = american_to_decimal(float(row["bet_odds"]))
            pnl = stake * (dec_odds - 1)
        else:
            pnl = -stake

        bankroll += pnl
        bankroll = max(bankroll, 0.01)  # floor at near-zero

        stake_history.append(stake)
        pnl_history.append(pnl)
        bankroll_history.append(bankroll)

    df["stake"]      = stake_history
    df["pnl"]        = pnl_history
    df["bankroll"]   = bankroll_history
    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df


# ── Main Backtest ─────────────────────────────────────────────────────────────

def run_backtest(model=None) -> dict:
    """
    Full backtest on the test season.
    Returns summary metrics and saves detailed CSV.
    """
    # Load feature matrix and always regenerate predictions fresh
    # (never use cached predictions_test.parquet — HOME_WIN may be missing)
    feature_path = PROC_DIR / "game_features.parquet"
    pred_path    = PROC_DIR / "predictions_test.parquet"

    if model is None:
        from src.model import NBAEnsemble
        model = NBAEnsemble().load()

    features = pd.read_parquet(feature_path)
    test_df  = features[features["SEASON"] == TEST_SEASON].copy()

    log.info(f"Test set: {len(test_df)} games | "
             f"HOME_WIN: {test_df['HOME_WIN'].value_counts().to_dict()}")

    # Generate predictions
    preds = model.predict_df(test_df)

    # Join outcomes and ELO_DIFF directly from the feature matrix by GAME_ID
    outcome_map = test_df.set_index("GAME_ID")["HOME_WIN"].to_dict()
    elo_map     = test_df.set_index("GAME_ID")["ELO_DIFF"].to_dict()
    preds["HOME_WIN"] = preds["GAME_ID"].map(outcome_map)
    preds["ELO_DIFF"] = preds["GAME_ID"].map(elo_map)

    log.info(f"HOME_WIN matched: {preds['HOME_WIN'].notna().sum()} / {len(preds)}")
    preds.to_parquet(pred_path, index=False)

    log.info(f"Backtesting {len(preds):,} games from {TEST_SEASON}")

    # Generate synthetic odds
    odds_dict = generate_synthetic_odds(preds)

    # Evaluate edge for each game
    bets = evaluate_slate(preds, odds_dict, min_edge=MIN_EDGE_PCT)

    # Add actual results
    if "HOME_WIN" in preds.columns:
        result_map = dict(zip(preds["GAME_ID"].astype(str), preds["HOME_WIN"]))
        bets["home_actually_won"] = bets["game_id"].map(result_map)

        # Was the bet correct?
        def bet_correct(row):
            if not row["has_edge"] or pd.isna(row["home_actually_won"]):
                return np.nan
            home_won = int(row["home_actually_won"])
            if row["bet_side"] == row["home_team"]:
                return 1 if home_won == 1 else 0
            else:
                return 1 if home_won == 0 else 0

        bets["result_correct"] = bets.apply(bet_correct, axis=1)
    else:
        bets["result_correct"] = np.nan

    # Bankroll simulation
    if bets["result_correct"].notna().any():
        sim = simulate_bankroll(bets)
    else:
        sim = bets[bets["has_edge"]].copy()

    # Save detailed results
    out_csv = LOG_DIR / f"backtest_{TEST_SEASON.replace('-','')}.csv"
    bets.to_csv(out_csv, index=False)
    log.info(f"Backtest results saved: {out_csv}")

    # Compute summary metrics
    flagged = bets[bets["has_edge"]].copy()
    decided = flagged[flagged["result_correct"].notna()].copy()

    if len(decided) == 0:
        log.warning("No decided bets in backtest — outcomes may be missing.")
        return {"error": "no_decided_bets"}

    wins       = (decided["result_correct"] == 1).sum()
    total_bets = len(decided)
    win_rate   = wins / total_bets

    # ROI (flat betting)
    avg_dec_odds = decided["bet_odds"].apply(american_to_decimal)
    gross_return = ((decided["result_correct"] == 1).astype(float) * (avg_dec_odds - 1)).sum()
    net_pnl_flat = gross_return - (decided["result_correct"] == 0).sum()
    roi_flat     = net_pnl_flat / total_bets * 100

    # Kelly ROI
    if "pnl" in sim.columns:
        kelly_roi = sim["pnl"].sum() / sim["stake"].sum() * 100
    else:
        kelly_roi = roi_flat

    # Calibration check
    brier = _brier_score(preds)

    # Sharpe (daily returns)
    if "pnl" in sim.columns and len(sim) > 5:
        daily = sim.groupby("game_date")["pnl"].sum() if "game_date" in sim.columns else sim["pnl"]
        sharpe = daily.mean() / (daily.std() + 1e-9) * np.sqrt(82)
    else:
        sharpe = 0.0

    summary = {
        "season":            TEST_SEASON,
        "total_games":       len(bets),
        "flagged_bets":      len(flagged),
        "flag_rate_pct":     round(len(flagged) / len(bets) * 100, 1),
        "decided_bets":      total_bets,
        "wins":              int(wins),
        "losses":            int(total_bets - wins),
        "win_rate":          round(win_rate, 4),
        "break_even_rate":   0.5238,   # at -110
        "roi_flat_pct":      round(roi_flat, 2),
        "roi_kelly_pct":     round(kelly_roi, 2),
        "avg_edge_pct":      round(decided["bet_edge_pct"].mean(), 2),
        "avg_kelly_units":   round(decided["kelly_units"].mean(), 3),
        "brier_score":       round(brier, 4),
        "sharpe_ratio":      round(sharpe, 3),
    }

    print("\n" + "="*55)
    print(f"  BACKTEST RESULTS — {TEST_SEASON}")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    print("="*55)

    return summary


def _brier_score(df: pd.DataFrame) -> float:
    if "HOME_WIN" in df.columns and "P_HOME_WIN" in df.columns:
        mask = df["HOME_WIN"].notna() & df["P_HOME_WIN"].notna()
        y = df.loc[mask, "HOME_WIN"].values
        p = df.loc[mask, "P_HOME_WIN"].values
        return float(np.mean((p - y) ** 2))
    return np.nan


def calibration_curve(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Compute calibration curve data.
    For each probability bin, compare model probability to actual win rate.
    """
    if "P_HOME_WIN" not in df.columns or "HOME_WIN" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["P_HOME_WIN", "HOME_WIN"]).copy()
    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["P_HOME_WIN"], bins=bins, include_lowest=True)

    cal = (
        df.groupby("bin", observed=True)
        .agg(
            n_games        = ("HOME_WIN", "count"),
            mean_pred_prob = ("P_HOME_WIN", "mean"),
            actual_win_rate= ("HOME_WIN", "mean"),
        )
        .reset_index()
    )
    cal["calibration_error"] = (cal["mean_pred_prob"] - cal["actual_win_rate"]).abs()
    return cal


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_backtest()
