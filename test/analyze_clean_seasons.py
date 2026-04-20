"""
analyze_clean_seasons.py — Filter analysis on the two clean held-out seasons only.
Run: python analyze_clean_seasons.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.edge import american_to_decimal, implied_prob_from_american, calculate_edge

LOG_DIR = Path("logs")

# Load only the two clean out-of-sample seasons
files = {
    "2024-25": LOG_DIR / "backtest_real_2025.csv",
    "2025-26": LOG_DIR / "backtest_real_2026.csv",
}

all_dfs = []
for season, path in files.items():
    if path.exists():
        df = pd.read_csv(path)
        df["season"] = season
        all_dfs.append(df)
    else:
        print(f"Missing: {path}")

if not all_dfs:
    print("Run backtest_real_odds.py for both seasons first.")
    sys.exit(1)

raw = pd.concat(all_dfs, ignore_index=True)
for col in ["model_prob_home","moneyline_home","moneyline_away","home_won"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

print(f"Loaded {len(raw)} games from {len(all_dfs)} clean out-of-sample seasons")
print()

def run_filter(df, min_edge, max_edge, min_odds, max_odds, underdogs_only):
    results = []
    for _, row in df.iterrows():
        p_home = row["model_prob_home"]
        h_odds = row["moneyline_home"]
        a_odds = row["moneyline_away"]
        if pd.isna(p_home) or pd.isna(h_odds) or pd.isna(a_odds):
            continue

        p_away = 1 - p_home
        fair_h, fair_a = implied_prob_from_american(h_odds, a_odds)
        edge_h = calculate_edge(p_home, fair_h)
        edge_a = calculate_edge(p_away, fair_a)

        if edge_h >= min_edge and edge_h >= edge_a:
            bet_odds, bet_edge, is_home = h_odds, edge_h, True
        elif edge_a >= min_edge:
            bet_odds, bet_edge, is_home = a_odds, edge_a, False
        else:
            continue

        if underdogs_only and bet_odds < 0:  continue
        if bet_odds > max_odds:              continue
        if bet_odds <= abs(min_odds):        continue
        if bet_edge > max_edge:              continue

        home_won = row["home_won"]
        if pd.isna(home_won): continue
        won = (is_home == (int(home_won) == 1))
        dec = american_to_decimal(bet_odds)
        pnl = (dec - 1) if won else -1.0

        results.append({
            "season":   row["season"],
            "is_home":  is_home,
            "bet_odds": bet_odds,
            "bet_edge": bet_edge,
            "won":      won,
            "pnl":      pnl,
        })

    return pd.DataFrame(results)


configs = [
    ("Current (12%, +141–+500, <30%)",          12, 30,  -140, 9999, True),
    ("Edge 10%",                                 10, 30,  -140, 9999, True),
    ("Edge 8%",                                   8, 30,  -140, 9999, True),
    ("Edge 15%",                                 15, 30,  -140, 9999, True),
    ("Edge 18%",                                 18, 30,  -140, 9999, True),
    ("No max_edge cap",                          12, 999, -140, 9999, True),
    ("Max_odds +300 only",                       12, 30,  -140,  300, True),
    ("Away bets only",                           12, 30,  -140, 9999, True),  # will filter below
    ("Edge 12%, odds +141 to +300",              12, 30,  -140,  300, True),
    ("Edge 15%, odds +141 to +500",              15, 30,  -140, 9999, True),
]

print(f"{'Config':42s} {'Bets':6s} {'Bets/yr':8s} {'WR':7s} {'ROI':8s} {'P&L':8s}")
print("-" * 82)

for i, (label, min_e, max_e, min_o, max_o, dogs_only) in enumerate(configs):
    df = run_filter(raw, min_e, max_e, min_o, max_o, dogs_only)

    # Special: away only
    if label == "Away bets only":
        df = df[~df["is_home"]]
        label = "Away bets only (12%)"

    if df.empty:
        print(f"  {label:40s}  no bets")
        continue

    total = len(df)
    wr    = df["won"].mean()
    roi   = df["pnl"].sum() / total * 100
    pnl   = df["pnl"].sum()
    bpy   = total / len(all_dfs)

    # Per season breakdown
    per_season = []
    for s in ["2024-25", "2025-26"]:
        sub = df[df["season"] == s]
        if len(sub):
            s_roi = sub["pnl"].sum() / len(sub) * 100
            per_season.append(f"{s[-2:]}:{s_roi:+.0f}%")

    detail = "  " + "  ".join(per_season)
    print(f"  {label:40s} {total:5d}  {bpy:5.0f}/yr  {wr:.1%}  {roi:+6.1f}%  {pnl:+6.1f}u{detail}")

print("-" * 82)
print()
print("Focus on ROI and per-season consistency.")
print("Best config = highest ROI with both seasons positive.")