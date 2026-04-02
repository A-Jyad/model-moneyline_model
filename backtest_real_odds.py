"""
backtest_real_odds.py — Backtest using real historical moneyline odds.

Data file: nba_2008-2025.csv
  - season column uses END year (e.g. season=2023 = 2022-23 NBA season)
  - Teams use short abbreviations: gs=GSW, sa=SAS, no=NOP, ny=NYK, utah=UTA, wsh=WAS
  - Odds available: 2008-2022 seasons (fully), 2023 (partial)
  - 2024-25 has NO odds data in this file

Best backtest target: season=2023 (2022-23 NBA season) — fully covered, out-of-sample

Usage:
    python backtest_real_odds.py                        # default: 2022-23 season
    python backtest_real_odds.py --file_season 2022     # 2021-22 season
    python backtest_real_odds.py --edge 8               # change edge threshold
    python backtest_real_odds.py --all                  # run all available seasons
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from config.settings import PROC_DIR, LOG_DIR
from src.edge import (
    american_to_decimal, implied_prob_from_american,
    calculate_edge, kelly_fraction_bet, expected_value
)

# ── Abbreviation normalisation ────────────────────────────────────────────────
# Maps this file's short codes -> standard NBA abbreviations used by the model
ABBR_MAP = {
    "GS":   "GSW",   # Golden State Warriors
    "SA":   "SAS",   # San Antonio Spurs
    "NO":   "NOP",   # New Orleans Pelicans
    "NY":   "NYK",   # New York Knicks
    "UTAH": "UTA",   # Utah Jazz
    "WSH":  "WAS",   # Washington Wizards
}

# Season in file (end year) -> NBA season string used by model
FILE_SEASON_TO_MODEL = {
    2019: "2018-19",
    2020: "2019-20",
    2021: "2020-21",
    2022: "2021-22",
    2023: "2022-23",
    2024: "2023-24",
    2025: "2024-25",
    2026: "2025-26",
}


def normalise_abbr(abbr: str) -> str:
    """Convert file abbreviation to standard NBA abbreviation."""
    upper = str(abbr).upper()
    return ABBR_MAP.get(upper, upper)


def load_odds(csv_path: str, file_season: int) -> pd.DataFrame:
    """Load and filter odds for a given file season."""
    df = pd.read_csv(csv_path)
    # Auto-detect format
    if 'home_ml' in df.columns:
        df = df.rename(columns={'home_ml': 'moneyline_home', 'away_ml': 'moneyline_away'})
        if 'home_abbr' in df.columns:
            df = df.rename(columns={'home_abbr': 'home_team_norm', 'away_abbr': 'away_team_norm'})
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['game_date_str'] = df['game_date'].dt.strftime('%Y-%m-%d')
        return df
    if 'season' in df.columns:
        df = df[df['season'] == file_season].copy()
    else:
        # No season column — use all rows (already filtered by date range during scrape)
        pass
    df = df[df['regular'] == True].copy()        # regular season only
    df = df.dropna(subset=['moneyline_home', 'moneyline_away'])

    if df.empty:
        raise ValueError(f"No odds data for season {file_season}. "
                         f"Seasons with full odds: 2008-2022.")

    # Parse date (format: DD/MM/YYYY)
    df['game_date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['game_date_str'] = df['game_date'].dt.strftime('%Y-%m-%d')

    # Normalise abbreviations
    df['home_abbr'] = df['home'].apply(normalise_abbr)
    df['away_abbr'] = df['away'].apply(normalise_abbr)

    print(f"Odds loaded: {len(df)} regular season games "
          f"(file season {file_season} = {FILE_SEASON_TO_MODEL.get(file_season, '?')})")
    return df


def load_predictions(model_season: str) -> pd.DataFrame:
    """Load model predictions for a given NBA season string."""
    feature_path = PROC_DIR / "game_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError("Feature matrix not found. Run: python run_pipeline.py --features")

    from src.model import NBAEnsemble
    model = NBAEnsemble().load()

    features = pd.read_parquet(feature_path)
    season_df = features[features['SEASON'] == model_season].copy()

    if season_df.empty:
        raise ValueError(f"No games for season {model_season} in feature matrix.\n"
                         f"Available: {features['SEASON'].unique().tolist()}")

    preds = model.predict_df(season_df)

    # Join actual outcomes
    outcome_map = season_df.set_index('GAME_ID')['HOME_WIN'].to_dict()
    preds['HOME_WIN'] = preds['GAME_ID'].map(outcome_map)

    # Normalise date for joining
    preds['game_date_str'] = pd.to_datetime(preds['GAME_DATE']).dt.strftime('%Y-%m-%d')
    preds['home_abbr'] = preds['HOME_TEAM_ABBREVIATION']
    preds['away_abbr'] = preds['AWAY_TEAM_ABBREVIATION']

    print(f"Predictions: {len(preds)} games for {model_season}")
    return preds


def join_and_evaluate(preds: pd.DataFrame,
                       odds: pd.DataFrame,
                       min_edge: float) -> pd.DataFrame:
    """Join predictions to odds and evaluate edge + results."""
    odds = odds.copy()

    # Ensure game_date_str
    if 'game_date_str' not in odds.columns:
        odds['game_date_str'] = pd.to_datetime(odds['game_date']).dt.strftime('%Y-%m-%d')

    # Normalise team abbreviation columns
    # SBR format: home_team_norm / away_team_norm
    # Old format: home_abbr / away_abbr
    # Fallback: map full names via SBR_TO_ABB
    if 'home_team_norm' in odds.columns:
        odds = odds.rename(columns={'home_team_norm': 'home_abbr',
                                     'away_team_norm': 'away_abbr'})
    if 'home_abbr' not in odds.columns:
        try:
            from src.sbr_scraper import SBR_TO_ABB
            odds['home_abbr'] = odds['home_team'].map(SBR_TO_ABB).fillna(
                odds['home_team'].str[:3].str.upper())
            odds['away_abbr'] = odds['away_team'].map(SBR_TO_ABB).fillna(
                odds['away_team'].str[:3].str.upper())
        except Exception:
            odds['home_abbr'] = odds.get('home_team', odds.get('home', ''))
            odds['away_abbr'] = odds.get('away_team', odds.get('away', ''))

    # Normalise odds column names
    if 'home_ml' in odds.columns and 'moneyline_home' not in odds.columns:
        odds = odds.rename(columns={'home_ml': 'moneyline_home',
                                     'away_ml': 'moneyline_away'})

    if 'moneyline_home' not in odds.columns:
        print(f"ERROR: Cannot find moneyline columns. Available: {list(odds.columns)}")
        return pd.DataFrame()

    merge_cols = ['game_date_str', 'home_abbr', 'away_abbr',
                  'moneyline_home', 'moneyline_away']
    odds_clean = odds[merge_cols].drop_duplicates(
        subset=['game_date_str', 'home_abbr', 'away_abbr'])

    print(f"Odds: {len(odds_clean)} games | Preds: {len(preds)} games")
    print(f"Sample odds dates: {odds_clean['game_date_str'].head(3).tolist()}")
    print(f"Sample preds dates: {preds['game_date_str'].head(3).tolist()}")
    print(f"Sample odds teams: {odds_clean[['home_abbr','away_abbr']].head(3).to_dict('records')}")
    print(f"Sample preds teams: {preds[['home_abbr','away_abbr']].head(3).to_dict('records')}")

    merged = preds.merge(
        odds_clean,
        on=['game_date_str', 'home_abbr', 'away_abbr'],
        how='inner'
    )
    match_rate = len(merged) / len(preds) * 100
    print(f"Matched: {len(merged)} / {len(preds)} games ({match_rate:.1f}%)")

    if merged.empty:
        return pd.DataFrame()

    from src.edge import evaluate_game as _evaluate_game
    results = []
    for _, row in merged.iterrows():
        p_home = float(row['P_HOME_WIN'])
        h_odds = float(row['moneyline_home'])
        a_odds = float(row['moneyline_away'])

        # Use evaluate_game which applies all configured filters
        # (underdogs_only, max_odds, min_odds, max_edge from settings.py)
        ev_result = _evaluate_game(
            home_team=row['home_abbr'],
            away_team=row['away_abbr'],
            model_prob_home=p_home,
            home_american_odds=h_odds,
            away_american_odds=a_odds,
            min_edge=min_edge,
        )

        has_edge  = ev_result['has_edge']
        bet_side  = ev_result.get('bet_side')
        bet_edge  = ev_result.get('bet_edge_pct')
        bet_odds  = ev_result.get('bet_odds')
        bet_prob  = ev_result.get('bet_prob')
        kelly     = ev_result.get('kelly_units', 0.0)
        ev        = ev_result.get('expected_value', 0.0)

        # Result
        home_won = row.get('HOME_WIN', np.nan)
        correct  = np.nan
        pnl      = np.nan
        if has_edge and not pd.isna(home_won):
            correct = 1 if (bet_side == row['home_abbr']) == (home_won == 1) else 0
            pnl = (american_to_decimal(bet_odds) - 1) if correct == 1 else -1.0

        results.append({
            'game_date':       row['game_date_str'],
            'home_team':       row['home_abbr'],
            'away_team':       row['away_abbr'],
            'model_prob_home': round(p_home, 4),
            'moneyline_home':  h_odds,
            'moneyline_away':  a_odds,
            'has_edge':        has_edge,
            'bet_side':        bet_side,
            'bet_edge_pct':    round(bet_edge, 2) if bet_edge else None,
            'bet_odds':        bet_odds,
            'kelly_frac':      round(float(kelly), 4) if kelly else 0.0,
            'expected_value':  round(float(ev), 4) if ev else None,
            'home_won':        home_won,
            'result_correct':  correct,
            'pnl_per_unit':    round(pnl, 4) if not pd.isna(pnl) else None,
        })

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame, season_label: str, min_edge: float):
    """Print backtest summary report."""
    flagged = df[df['has_edge']].copy()
    decided = flagged[flagged['result_correct'].notna()].copy()

    if decided.empty:
        print("No decided bets.")
        return

    wins   = (decided['result_correct'] == 1).sum()
    total  = len(decided)
    wr     = wins / total
    pnl    = decided['pnl_per_unit'].sum()
    roi    = pnl / total * 100
    avg_dec = decided['bet_odds'].apply(american_to_decimal).mean()
    be_rate = 1 / avg_dec * 100

    print()
    print("=" * 60)
    print(f"  REAL ODDS BACKTEST — {season_label}  (edge >= {min_edge}%)")
    print("=" * 60)
    print(f"  Total matched games  : {len(df)}")
    print(f"  Flagged bets         : {total} ({total/len(df)*100:.1f}%)")
    print(f"  Wins / Losses        : {int(wins)} / {int(total - wins)}")
    print(f"  Win rate             : {wr:.1%}  (break-even: {be_rate:.1f}%)")
    print(f"  ROI flat             : {roi:+.2f}%")
    print(f"  Total P&L            : {pnl:+.2f} units on {total} bets")
    print(f"  Avg edge             : {decided['bet_edge_pct'].mean():.2f}%")

    # Home vs Away
    home_b = decided[decided['bet_side'] == decided['home_team']]
    away_b = decided[decided['bet_side'] != decided['home_team']]
    print()
    if len(home_b):
        h_roi = home_b['pnl_per_unit'].sum() / len(home_b) * 100
        print(f"  Home bets: {len(home_b):3d} | WR: {(home_b['result_correct']==1).mean():.1%} | ROI: {h_roi:+.1f}%")
    if len(away_b):
        a_roi = away_b['pnl_per_unit'].sum() / len(away_b) * 100
        print(f"  Away bets: {len(away_b):3d} | WR: {(away_b['result_correct']==1).mean():.1%} | ROI: {a_roi:+.1f}%")

    # Threshold sweep
    print()
    print(f"  {'Threshold':10s} {'Bets':6s} {'Win%':7s} {'ROI':8s}")
    print("  " + "-" * 36)
    for thresh in [4, 6, 8, 10, 12, 15]:
        sub = df[
            df['has_edge'] &
            (df['bet_edge_pct'] >= thresh) &
            df['result_correct'].notna()
        ]
        if len(sub) < 5:
            continue
        s_wr  = (sub['result_correct'] == 1).mean()
        s_roi = sub['pnl_per_unit'].sum() / len(sub) * 100
        print(f"  >= {thresh:2d}%      {len(sub):6d} {s_wr*100:5.1f}%   {s_roi:+6.1f}%")

    print("=" * 60)


def run_backtest(csv_path: str, file_season: int,
                 min_edge: float = 10.0) -> pd.DataFrame:
    model_season = FILE_SEASON_TO_MODEL.get(file_season)
    if not model_season:
        raise ValueError(f"Unknown file season: {file_season}")

    odds  = load_odds(csv_path, file_season)
    preds = load_predictions(model_season)
    df    = join_and_evaluate(preds, odds, min_edge)

    if df.empty:
        print("No matched games. Check date/team alignment.")
        return df

    # Save
    out = LOG_DIR / f"backtest_real_{file_season}.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")

    print_summary(df, model_season, min_edge)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default=None, help="Path to SBR odds CSV (auto-detected if omitted)")
    parser.add_argument("--file_season", type=int, default=2023,
                        help="Season end-year in file (2023=2022-23). Default: 2023")
    parser.add_argument("--edge",        type=float, default=10.0)
    parser.add_argument("--all",         action="store_true",
                        help="Run all seasons with full odds coverage")
    args = parser.parse_args()

    if args.all:
        from pathlib import Path
        all_results = []
        for fs, model_s in FILE_SEASON_TO_MODEL.items():
            slug = f"{model_s[:4]}_{model_s[5:7]}"
            csv  = f"data/odds/sbr_{slug}.csv"
            if not Path(csv).exists():
                print(f"  Skipping {model_s}: {csv} not found — scrape first")
                continue
            print(f"\n{'='*60}")
            print(f"  {model_s}  ({csv})")
            print(f"{'='*60}")
            try:
                df = run_backtest(csv, fs, args.edge)
                if df is not None and not df.empty:
                    decided = df[df['has_edge'] & df['result_correct'].notna()]
                    if len(decided):
                        roi = decided['pnl_per_unit'].sum()/len(decided)*100
                        wr  = (decided['result_correct']==1).mean()
                        all_results.append({'season': model_s, 'bets': len(decided),
                                            'win_rate': wr, 'roi': roi})
            except Exception as e:
                print(f"  Error: {e}")

        if all_results:
            print(f"\n{'='*60}")
            print("  MULTI-SEASON SUMMARY")
            print(f"{'='*60}")
            import pandas as pd
            summary = pd.DataFrame(all_results)
            total_bets = summary['bets'].sum()
            combined_roi = (summary['roi'] * summary['bets']).sum() / total_bets
            print(f"  {'Season':10s} {'Bets':6s} {'Win%':8s} {'ROI':8s}")
            print(f"  {'-'*36}")
            for _, r in summary.iterrows():
                print(f"  {r['season']:10s} {r['bets']:6d} {r['win_rate']:.1%}    {r['roi']:+.1f}%")
            print(f"  {'-'*36}")
            print(f"  {'Combined':10s} {total_bets:6d}            {combined_roi:+.1f}%")
            print(f"{'='*60}")
    else:
        # Auto-detect CSV path from file_season if not provided
        if args.csv is None:
            from pathlib import Path
            model_s = FILE_SEASON_TO_MODEL.get(args.file_season, "")
            slug    = f"{model_s[:4]}_{model_s[5:7]}" if model_s else str(args.file_season)
            args.csv = f"data/odds/sbr_{slug}.csv"
            if not Path(args.csv).exists():
                print(f"No odds file found at {args.csv}")
                print(f"Scrape first: python src/sbr_scraper.py --from YYYY-MM-DD --to YYYY-MM-DD --out {args.csv} --ml-only")
                exit(1)
        run_backtest(args.csv, args.file_season, args.edge)