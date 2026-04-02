#!/usr/bin/env python3
"""
run_pipeline.py — Master runner for the NBA Prediction Model.

Commands:
  --fetch      Incrementally scrape all season game logs (resumes from cache)
  --features   Build feature engineering pipeline
  --train      Train + calibrate ensemble model
  --backtest   Run historical backtest on test season
  --predict    Generate predictions for today's slate
  --dashboard  Open interactive dashboard
  --update     Incremental nightly update
  --track      Show performance report
  --export     Export analytics CSVs
  --all        Run full pipeline (fetch->features->train->backtest->predict)
  --status     Show cache/model status

Examples:
  python run_pipeline.py --fetch
  python run_pipeline.py --all
  python run_pipeline.py --predict --date 2025-04-01
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

# ── Auto-create all required directories before anything else ─────────────────
# Prevents FileNotFoundError on first run (no logs/, data/ folders yet)
_BASE = Path(__file__).resolve().parent
for _d in ["logs", "data/cache", "data/raw", "data/processed", "models"]:
    (_BASE / _d).mkdir(parents=True, exist_ok=True)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_BASE / "logs" / "pipeline.log", mode="a"),
    ]
)
log = logging.getLogger("pipeline")

# ── Path setup ────────────────────────────────────────────────────────────────
# Insert project root so 'config' and 'src' are always importable
# regardless of where the script is run from
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))
if str(_BASE.parent) not in sys.path:
    sys.path.insert(0, str(_BASE.parent))

from config.settings import (
    CACHE_DIR, RAW_DIR, PROC_DIR, MODEL_DIR, LOG_DIR, SEASONS, TEST_SEASON
)


# ── Step functions ─────────────────────────────────────────────────────────────

def step_fetch(args):
    log.info("="*55)
    log.info("STEP 1: DATA FETCH (incremental)")
    log.info("="*55)
    from src.scraper import fetch_all_data
    from src.predict import get_current_season
    # Use --season arg if provided, otherwise auto-detect from today's date
    season = getattr(args, "season", None) or get_current_season()
    log.info(f"Current season: {season}")
    results = fetch_all_data(current_season=season)
    if results["game_logs"].empty:
        log.error("No game log data fetched. Check network connection.")
        return False
    return True


def step_features(args):
    log.info("="*55)
    log.info("STEP 2: FEATURE ENGINEERING")
    log.info("="*55)
    from src.features import run_feature_pipeline
    try:
        df = run_feature_pipeline()
        log.info(f"Feature matrix: {len(df):,} games x {df.shape[1]} features")
        return True
    except FileNotFoundError as e:
        log.error(f"{e}")
        return False


def step_train(args):
    log.info("="*55)
    log.info("STEP 3: MODEL TRAINING")
    log.info("="*55)
    from src.model import train_and_save
    try:
        train_and_save()
        log.info("Models trained and saved.")
        return True
    except Exception as e:
        log.exception(f"Training failed: {e}")
        return False


def step_backtest(args):
    log.info("="*55)
    log.info("STEP 4: BACKTEST")
    log.info("="*55)
    from src.backtest import run_backtest
    try:
        summary = run_backtest()
        if "error" not in summary:
            roi = summary.get("roi_flat_pct", 0)
            wr  = summary.get("win_rate", 0)
            log.info(f"Backtest complete - ROI: {roi:+.1f}%  Win rate: {wr:.1%}")
        return True
    except Exception as e:
        log.exception(f"Backtest failed: {e}")
        return False


def step_predict(args):
    log.info("="*55)
    log.info("STEP 5: LIVE PREDICTIONS")
    log.info("="*55)
    from src.predict import predict_today, print_slate_report, get_current_season
    target_date = getattr(args, "date", None)
    season = getattr(args, "season", None) or get_current_season(target_date)
    log.info(f"Season auto-detected: {season}")

    # Parse manual odds if provided
    # Format: "HOME1:ML,AWAY1:ML;HOME2:ML,AWAY2:ML"
    # Example: "CHA:-180,PHX:+155;GSW:+220,CLE:-270"
    odds_dict = {}
    odds_str = getattr(args, "odds", None)
    if odds_str:
        try:
            for game in odds_str.split(";"):
                parts = game.strip().split(",")
                if len(parts) == 2:
                    home_team, home_ml = parts[0].strip().split(":")
                    away_team, away_ml = parts[1].strip().split(":")
                    odds_dict[(home_team.strip().upper(), away_team.strip().upper())] = {
                        "home": int(home_ml), "away": int(away_ml)
                    }
            if odds_dict:
                log.info(f"Manual odds loaded: {len(odds_dict)} games")
        except Exception as e:
            log.warning(f"Could not parse --odds argument: {e}")

    try:
        preds = predict_today(target_date=target_date, season=season,
                              odds_dict=odds_dict if odds_dict else None)
        print_slate_report(preds)
        return True
    except Exception as e:
        log.exception(f"Prediction failed: {e}")
        return False


def step_status(args):
    print("\n" + "="*55)
    print("  NBA PREDICTOR - STATUS")
    print("="*55)

    print("\n  Cache Status:")
    checkpoints = list(CACHE_DIR.glob("checkpoint_*.json"))
    game_caches = list(CACHE_DIR.glob("games_*.json"))
    print(f"    Checkpoint files : {len(checkpoints)}")
    print(f"    Game cache files : {len(game_caches)}")

    for ckpt in sorted(checkpoints):
        try:
            with open(ckpt) as f:
                data = json.load(f)
            season_name = ckpt.stem.replace("checkpoint_", "")
            last_date   = data.get("last_date", "never")
            n_games     = data.get("games_fetched", 0)
            print(f"    {season_name:30s}: last={last_date}  n={n_games}")
        except Exception:
            pass

    import pandas as pd
    print("\n  Raw Data Files:")
    raw_files = {
        "all_game_logs.parquet":       "Game logs",
        "advanced_team_stats.parquet": "Advanced stats",
        "schedule_202425.parquet":     "Schedule 2024-25",
        "injury_report.parquet":       "Injury report",
    }
    any_raw = False
    for fname, label in raw_files.items():
        fpath = RAW_DIR / fname
        if fpath.exists():
            df_raw = pd.read_parquet(fpath)
            print(f"    {label:25s}: {len(df_raw):,} rows  ({fname})")
            any_raw = True
        else:
            print(f"    {label:25s}: NOT FOUND")
    if not any_raw:
        print("    No raw data yet — run --fetch")

    feat_path = PROC_DIR / "game_features.parquet"
    if feat_path.exists():
        import pandas as pd
        df = pd.read_parquet(feat_path)
        print(f"  Feature Matrix: {len(df):,} games x {df.shape[1]} columns")
    else:
        print("  Feature Matrix: NOT FOUND - run --features")

    model_files = list(MODEL_DIR.glob("*.pkl"))
    if model_files:
        print(f"  Models: {len(model_files)} files saved")
    else:
        print("  Models: NOT TRAINED - run --train")

    bet_log = LOG_DIR / "bet_log.csv"
    try:
        if bet_log.exists() and bet_log.stat().st_size > 10:
            from src.edge import bet_log_summary
            summary = bet_log_summary()
            print(f"\n  Bet Log: {summary['total_bets']} bets | "
                  f"ROI: {summary.get('roi_pct', 0):+.1f}% | "
                  f"Win rate: {summary.get('win_rate', 0):.1%}")
        else:
            print("\n  Bet Log: No bets recorded yet")
    except Exception:
        print("\n  Bet Log: No bets recorded yet")

    print("="*55 + "\n")


def step_dashboard(args):
    from src.dashboard import run_dashboard
    run_dashboard(
        target_date=getattr(args, "date", None),
        season=getattr(args, "season", None),
    )
    return True


def step_update(args):
    log.info("="*55)
    log.info("STEP: INCREMENTAL UPDATE")
    log.info("="*55)
    from src.updater import run_incremental_update
    summary = run_incremental_update(season=getattr(args, "season", None))
    if summary.get("alerts"):
        for alert in summary["alerts"]:
            log.warning(f"  ! {alert}")
    log.info(f"Update done. New games: {summary.get('new_games', 0)}")
    return True


def step_track(args):
    from src.tracker import print_performance_report
    print_performance_report()
    return True


def step_export(args):
    from src.tracker import export_full_report
    export_full_report()
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NBA Moneyline Prediction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fetch",     action="store_true", help="Fetch game data (incremental)")
    parser.add_argument("--features",  action="store_true", help="Build feature matrix")
    parser.add_argument("--train",     action="store_true", help="Train models")
    parser.add_argument("--backtest",  action="store_true", help="Run backtest")
    parser.add_argument("--predict",   action="store_true", help="Predict today's slate")
    parser.add_argument("--dashboard", action="store_true", help="Open interactive dashboard")
    parser.add_argument("--update",    action="store_true", help="Incremental update (nightly)")
    parser.add_argument("--track",     action="store_true", help="Show performance report")
    parser.add_argument("--export",    action="store_true", help="Export analytics CSVs")
    parser.add_argument("--all",       action="store_true", help="Run full pipeline")
    parser.add_argument("--status",    action="store_true", help="Show status")
    parser.add_argument("--date",      default=None,        help="Date for prediction (YYYY-MM-DD)")
    parser.add_argument("--season",    default=None,        help="Season string (default: auto-detect)")
    parser.add_argument("--edge",      type=float, default=None, help="Override min edge %")

    args = parser.parse_args()

    if not any([args.fetch, args.features, args.train,
                args.backtest, args.predict, args.all,
                args.status, args.dashboard, args.update,
                args.track, args.export]):
        parser.print_help()
        return

    if args.status:
        step_status(args)
        return

    if args.all:
        steps = [step_fetch, step_features, step_train, step_backtest, step_predict]
        for i, step in enumerate(steps, 1):
            ok = step(args)
            if not ok:
                log.error(f"Pipeline aborted at step {i}.")
                sys.exit(1)
        log.info("Full pipeline complete.")
        return

    if args.fetch:
        step_fetch(args)
    if args.features:
        step_features(args)
    if args.train:
        step_train(args)
    if args.backtest:
        step_backtest(args)
    if args.predict:
        step_predict(args)
    if args.dashboard:
        step_dashboard(args)
    if args.update:
        step_update(args)
    if args.track:
        step_track(args)
    if args.export:
        step_export(args)


if __name__ == "__main__":
    main()