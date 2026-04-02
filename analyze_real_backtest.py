"""
analyze_real_backtest.py — Deep analysis of real odds backtest results.
Run: python analyze_real_backtest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.edge import american_to_decimal

LOG_DIR = Path(__file__).parent / "logs"
df = pd.read_csv(LOG_DIR / "backtest_real_2025.csv")
df = df[df['has_edge'] == True].copy()
df = df[df['result_correct'].notna()].copy()

df['result_correct'] = pd.to_numeric(df['result_correct'])
df['bet_odds']       = pd.to_numeric(df['bet_odds'])
df['bet_edge_pct']   = pd.to_numeric(df['bet_edge_pct'])
df['model_prob_home']= pd.to_numeric(df['model_prob_home'])
df['pnl_per_unit']   = pd.to_numeric(df['pnl_per_unit'])

df['dec_odds'] = df['bet_odds'].apply(american_to_decimal)
df['break_even'] = 1 / df['dec_odds']
df['is_home_bet'] = df['bet_side'] == df['home_team']
df['is_underdog']  = df['bet_odds'] > 0

print("=" * 65)
print("  REAL ODDS BACKTEST — DEEP ANALYSIS  (2024-25 season)")
print("=" * 65)

# 1. Odds range breakdown
print("\n1. PERFORMANCE BY ODDS RANGE")
print(f"   {'Odds Range':18s} {'Bets':6s} {'Win%':7s} {'BE%':7s} {'ROI':8s}")
print("   " + "-" * 50)
bins = [(-2000,-300),(-300,-200),(-200,-140),(-140,-110),(-110,100),(100,200),(200,500),(500,2000)]
labels = ['<-300','−300/−200','−200/−140','−140/−110','−110/EV','100/200','200/500','>+500']
for (lo, hi), lbl in zip(bins, labels):
    sub = df[(df['bet_odds'] >= lo) & (df['bet_odds'] < hi)]
    if len(sub) < 5: continue
    wr  = (sub['result_correct']==1).mean()
    be  = (1/sub['dec_odds']).mean()
    roi = sub['pnl_per_unit'].sum() / len(sub) * 100
    print(f"   {lbl:18s} {len(sub):6d} {wr*100:5.1f}%   {be*100:5.1f}%   {roi:+6.1f}%")

# 2. Favorite vs underdog
print("\n2. FAVORITE vs UNDERDOG")
fav = df[~df['is_underdog']]
dog = df[df['is_underdog']]
for label, sub in [("Favorites (negative odds)", fav), ("Underdogs (positive odds)", dog)]:
    if len(sub) == 0: continue
    wr  = (sub['result_correct']==1).mean()
    be  = (1/sub['dec_odds']).mean()
    roi = sub['pnl_per_unit'].sum() / len(sub) * 100
    print(f"   {label}: {len(sub)} bets | WR {wr:.1%} | BE {be:.1%} | ROI {roi:+.1f}%")

# 3. Edge size vs actual ROI
print("\n3. EDGE SIZE vs ACTUAL ROI")
print(f"   {'Edge bucket':15s} {'Bets':6s} {'Win%':7s} {'Avg odds':10s} {'ROI':8s}")
print("   " + "-" * 50)
for lo, hi in [(10,12),(12,15),(15,20),(20,30),(30,50),(50,100)]:
    sub = df[(df['bet_edge_pct']>=lo) & (df['bet_edge_pct']<hi)]
    if len(sub) < 5: continue
    wr  = (sub['result_correct']==1).mean()
    roi = sub['pnl_per_unit'].sum() / len(sub) * 100
    avg_odds = sub['bet_odds'].mean()
    print(f"   {lo}-{hi}%          {len(sub):6d} {wr*100:5.1f}%   {avg_odds:+7.0f}   {roi:+6.1f}%")

# 4. Model probability calibration check
print("\n4. MODEL CALIBRATION vs ACTUAL RESULTS")
print(f"   {'Model prob':12s} {'N':5s} {'Actual WR':11s} {'Diff':8s}")
print("   " + "-" * 40)
bins_p = [0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.8,1.0]
for i in range(len(bins_p)-1):
    lo, hi = bins_p[i], bins_p[i+1]
    # prob applies to the BET side, not necessarily home
    sub = df.copy()
    sub['bet_prob'] = pd.to_numeric(sub.get('bet_prob', sub['model_prob_home']), errors='coerce')
    sub = sub[(sub['bet_prob']>=lo) & (sub['bet_prob']<hi)]
    if len(sub) < 5: continue
    wr   = (sub['result_correct']==1).mean()
    avg  = sub['bet_prob'].mean()
    diff = wr - avg
    flag = " <<< overconfident" if diff < -0.05 else (" <<< underconfident" if diff > 0.05 else "")
    print(f"   {lo:.0%}-{hi:.0%}      {len(sub):5d}  {wr:.1%}      {diff:+.1%}{flag}")

# 5. Key finding
print()
print("=" * 65)
print("  KEY FINDING")
print("=" * 65)
total_roi = df['pnl_per_unit'].sum() / len(df) * 100
dog_roi   = dog['pnl_per_unit'].sum() / len(dog) * 100 if len(dog) else 0
fav_roi   = fav['pnl_per_unit'].sum() / len(fav) * 100 if len(fav) else 0

print(f"  Overall ROI:          {total_roi:+.2f}%  ({len(df)} bets)")
print(f"  Underdog bets ROI:    {dog_roi:+.2f}%  ({len(dog)} bets, {len(dog)/len(df):.0%} of all)")
print(f"  Favorite bets ROI:    {fav_roi:+.2f}%  ({len(fav)} bets, {len(fav)/len(df):.0%} of all)")
print()
print("  The model's edge, if any, comes from underdogs.")
print("  High win rate on favorites still loses to vig.")
print("  Filter: consider betting ONLY underdogs with edge >= 12%.")
print()

# Underdog only, edge 12%+
best = df[df['is_underdog'] & (df['bet_edge_pct'] >= 12)]
if len(best) >= 10:
    br  = (best['result_correct']==1).mean()
    boi = best['pnl_per_unit'].sum() / len(best) * 100
    be2 = (1/best['dec_odds']).mean()
    print(f"  Underdogs >= 12% edge: {len(best)} bets | WR {br:.1%} vs BE {be2:.1%} | ROI {boi:+.1f}%")
print("=" * 65)
