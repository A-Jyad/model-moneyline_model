"""
predict.py — Daily prediction script with manual odds input.

Usage:
    python predict.py
    python predict.py --odds "CHA:-180,PHX:+155;GSW:+220,CLE:-270"

Odds format: "HOME:ML,AWAY:ML;HOME:ML,AWAY:ML"
  - Separate games with semicolon ;
  - Home team first, then away team
  - Example: "LAL:-150,GSW:+130;BOS:-200,MIA:+170"
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)-10s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("predict")

from src.predict import predict_today, print_slate_report, get_current_season
from datetime import date

def parse_odds(odds_str: str) -> dict:
    """Parse odds string into dict keyed by (home_abbr, away_abbr)."""
    odds = {}
    if not odds_str:
        return odds
    for game in odds_str.split(";"):
        game = game.strip()
        if not game:
            continue
        parts = game.split(",")
        if len(parts) != 2:
            log.warning(f"Invalid game format: {game} (expected HOME:ML,AWAY:ML)")
            continue
        try:
            home_part = parts[0].strip()
            away_part = parts[1].strip()
            home_team, home_ml = home_part.rsplit(":", 1)
            away_team, away_ml = away_part.rsplit(":", 1)
            odds[(home_team.strip().upper(), away_team.strip().upper())] = {
                "home": int(home_ml.strip()),
                "away": int(away_ml.strip()),
            }
        except Exception as e:
            log.warning(f"Could not parse '{game}': {e}")
    return odds


def main():
    parser = argparse.ArgumentParser(description="NBA daily predictions")
    parser.add_argument("--date",   default=None, help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--season", default=None, help="Season string (default: auto-detect)")
    parser.add_argument("--edge",   type=float, default=None, help="Min edge %% override")
    parser.add_argument("--odds",   default=None,
                        help='Manual odds: "HOME:ML,AWAY:ML;HOME:ML,AWAY:ML"\n'
                             'Example: "CHA:-180,PHX:+155;GSW:+220,CLE:-270"')
    args = parser.parse_args()

    target_date = args.date or date.today().strftime("%Y-%m-%d")
    season      = args.season or get_current_season(target_date)
    odds_dict   = parse_odds(args.odds) if args.odds else None

    if odds_dict:
        log.info(f"Manual odds provided for {len(odds_dict)} games")
        for (h, a), v in odds_dict.items():
            h_str = f"+{v['home']}" if v['home'] > 0 else str(v['home'])
            a_str = f"+{v['away']}" if v['away'] > 0 else str(v['away'])
            log.info(f"  {h} {h_str} / {a} {a_str}")

    preds = predict_today(
        target_date=target_date,
        season=season,
        odds_dict=odds_dict,
        min_edge=args.edge,
    )
    print_slate_report(preds)


if __name__ == "__main__":
    main()