"""
sbr_scraper.py — Sportsbookreview NBA Scraper: Moneyline + Totals + Spreads
============================================================================
Scrapes historical closing moneylines, totals AND spreads from SBR.
No API key needed — free, runs locally.

Usage:
    python src/sbr_scraper.py --from 2024-10-22 --to 2025-04-13 --out data/odds/sbr_2024_25.csv
    python src/sbr_scraper.py --from 2022-10-18 --to 2023-06-12 --out data/odds/sbr_2022_23.csv

Requirements:
    pip install requests beautifulsoup4 pandas
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

URLS = {
    "totals":    "https://www.sportsbookreview.com/betting-odds/nba-basketball/totals/full-game/",
    "spread":    "https://www.sportsbookreview.com/betting-odds/nba-basketball/pointspread/full-game/",
    "moneyline": "https://www.sportsbookreview.com/betting-odds/nba-basketball/money-line/full-game/",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.sportsbookreview.com/",
}

BOOK_PRIORITY = ["pinnacle", "draftkings", "fanduel", "caesars", "betmgm", "bet365"]

SBR_TO_ABB = {
    "Atlanta Hawks":          "ATL", "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN", "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI", "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL", "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET", "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU", "Indiana Pacers":         "IND",
    "LA Clippers":            "LAC", "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL", "LA Lakers":              "LAL",
    "Memphis Grizzlies":      "MEM", "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP", "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC", "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI", "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS", "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA", "Washington Wizards":     "WAS",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def date_range(start: str, end: str):
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt  = datetime.strptime(end,   "%Y-%m-%d")
    while current <= end_dt:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def get_game_rows(html: str) -> list:
    soup   = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"id": "__NEXT_DATA__"})
    if not script:
        return []
    try:
        data = json.loads(script.string)
        return (
            data["props"]["pageProps"]
                .get("oddsTables", [{}])[0]
                .get("oddsTableModel", {})
                .get("gameRows", [])
        )
    except Exception:
        return []


def best_book(odds_views: list, fields: list) -> dict | None:
    book_odds = {}
    for view in odds_views:
        if view is None:
            continue
        book_name = view.get("sportsbook", "")
        current   = view.get("currentLine", {}) or {}
        values    = {f: current.get(f) for f in fields}
        if all(v is not None for v in values.values()):
            book_odds[book_name] = values

    for book in BOOK_PRIORITY:
        if book in book_odds:
            return {"source_book": book, **book_odds[book]}

    if book_odds:
        first_key = next(iter(book_odds))
        return {"source_book": first_key, **book_odds[first_key]}

    return None


# ── Moneyline scraper (NEW) ───────────────────────────────────────────────────

def fetch_moneylines(date: str, session: requests.Session) -> list[dict]:
    """
    Scrape closing moneylines for a single date from SBR.

    SBR moneyline page __NEXT_DATA__ structure:
      gameRow -> oddsViews[] -> currentLine -> {homeOdds, awayOdds}
    Home/away odds are American format (e.g. -150, +130).
    """
    try:
        resp = session.get(
            f"{URLS['moneyline']}?date={date}",
            headers=HEADERS, timeout=15
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [moneyline] request failed {date}: {e}")
        return []

    rows = []
    for event in get_game_rows(resp.text):
        try:
            gv        = event.get("gameView", {})
            home_team = gv.get("homeTeam", {}).get("fullName", "")
            away_team = gv.get("awayTeam", {}).get("fullName", "")
            if not home_team or not away_team:
                continue

            # SBR moneyline fields: homeOdds / awayOdds in currentLine
            result = best_book(
                event.get("oddsViews", []),
                ["homeOdds", "awayOdds"]
            )
            if result is None:
                continue

            rows.append({
                "game_date":    date,
                "home_team":    home_team,
                "away_team":    away_team,
                "home_ml":      int(result["homeOdds"]),
                "away_ml":      int(result["awayOdds"]),
                "ml_book":      result["source_book"],
                # Map to standard abbreviations
                "home_abbr":    SBR_TO_ABB.get(home_team, home_team[:3].upper()),
                "away_abbr":    SBR_TO_ABB.get(away_team, away_team[:3].upper()),
            })
        except Exception:
            continue

    return rows


# ── Totals scraper (unchanged) ────────────────────────────────────────────────

def fetch_totals(date: str, session: requests.Session) -> list[dict]:
    try:
        resp = session.get(f"{URLS['totals']}?date={date}", headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [totals] request failed {date}: {e}")
        return []

    rows = []
    for event in get_game_rows(resp.text):
        try:
            gv        = event.get("gameView", {})
            home_team = gv.get("homeTeam", {}).get("fullName", "")
            away_team = gv.get("awayTeam", {}).get("fullName", "")
            if not home_team or not away_team:
                continue

            result = best_book(event.get("oddsViews", []), ["total", "overOdds", "underOdds"])
            if result is None:
                continue

            rows.append({
                "game_date":    date,
                "start_time":   gv.get("startDate", date),
                "home_team":    home_team,
                "away_team":    away_team,
                "market_total": float(result["total"]),
                "over_odds":    int(result["overOdds"]),
                "under_odds":   int(result["underOdds"]),
                "total_book":   result["source_book"],
            })
        except Exception:
            continue

    return rows


# ── Spreads scraper (unchanged) ───────────────────────────────────────────────

def fetch_spreads(date: str, session: requests.Session) -> list[dict]:
    try:
        resp = session.get(f"{URLS['spread']}?date={date}", headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [spread] request failed {date}: {e}")
        return []

    rows = []
    for event in get_game_rows(resp.text):
        try:
            gv        = event.get("gameView", {})
            home_team = gv.get("homeTeam", {}).get("fullName", "")
            away_team = gv.get("awayTeam", {}).get("fullName", "")
            if not home_team or not away_team:
                continue

            result = best_book(event.get("oddsViews", []), ["homeSpread", "homeOdds", "awayOdds"])
            if result is None:
                continue

            rows.append({
                "game_date":     date,
                "home_team":     home_team,
                "away_team":     away_team,
                "market_spread": float(result["homeSpread"]),
                "home_odds":     int(result["homeOdds"]),
                "away_odds":     int(result["awayOdds"]),
                "spread_book":   result["source_book"],
            })
        except Exception:
            continue

    return rows


# ── Full scrape: moneyline + totals + spreads ─────────────────────────────────

def scrape_range(date_from: str, date_to: str, delay: float = 2.0) -> pd.DataFrame:
    """
    Scrape moneylines, totals AND spreads for every date in range.
    Merges all three into one row per game.
    delay: seconds between date batches — keep >= 2.0 to avoid blocks.
    """
    session      = requests.Session()
    all_ml       = []
    all_totals   = []
    all_spreads  = []
    dates        = list(date_range(date_from, date_to))

    print(f"Scraping {len(dates)} dates ({date_from} → {date_to})...")
    print(f"Fetching: moneyline + totals + spreads per date\n")

    for i, date in enumerate(dates):
        ml      = fetch_moneylines(date, session)
        time.sleep(0.4)
        totals  = fetch_totals(date, session)
        time.sleep(0.4)
        spreads = fetch_spreads(date, session)

        if ml:      all_ml.extend(ml)
        if totals:  all_totals.extend(totals)
        if spreads: all_spreads.extend(spreads)

        if ml or totals or spreads:
            print(f"  {date}  →  {len(ml)} ML  |  {len(totals)} totals  |  {len(spreads)} spreads")
        else:
            print(f"  {date}  →  no games")

        # Progress save every 30 dates
        if (i + 1) % 30 == 0 and all_ml:
            _save_checkpoint(all_ml, all_totals, all_spreads, date_from, date_to)
            print(f"  [checkpoint saved at {date}]")

        if i < len(dates) - 1:
            time.sleep(delay)

    if not all_ml:
        print("\nNo moneylines scraped. Check network access to sportsbookreview.com")
        return pd.DataFrame()

    # ── Build moneyline base ──────────────────────────────────
    df = pd.DataFrame(all_ml)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ── Merge totals ──────────────────────────────────────────
    if all_totals:
        df_totals = pd.DataFrame(all_totals)
        df_totals["game_date"] = pd.to_datetime(df_totals["game_date"])
        df = df.merge(
            df_totals[["game_date", "home_team", "away_team",
                        "market_total", "over_odds", "under_odds", "total_book"]],
            on=["game_date", "home_team", "away_team"],
            how="left",
        )

    # ── Merge spreads ─────────────────────────────────────────
    if all_spreads:
        df_spreads = pd.DataFrame(all_spreads)
        df_spreads["game_date"] = pd.to_datetime(df_spreads["game_date"])
        df = df.merge(
            df_spreads[["game_date", "home_team", "away_team",
                         "market_spread", "spread_book"]],
            on=["game_date", "home_team", "away_team"],
            how="left",
        )

    df = df.sort_values("game_date").reset_index(drop=True)

    # ── Implied score features (from total + spread) ──────────
    if "market_total" in df.columns and "market_spread" in df.columns:
        has_both = df["market_total"].notna() & df["market_spread"].notna()
        df.loc[has_both, "implied_home_score"] = (
            df.loc[has_both, "market_total"] - df.loc[has_both, "market_spread"]
        ) / 2
        df.loc[has_both, "implied_away_score"] = (
            df.loc[has_both, "market_total"] + df.loc[has_both, "market_spread"]
        ) / 2
        df.loc[has_both, "implied_score_diff"] = (
            df.loc[has_both, "implied_home_score"] - df.loc[has_both, "implied_away_score"]
        )

    ml_cov     = df["home_ml"].notna().mean()
    total_cov  = df["market_total"].notna().mean() if "market_total" in df.columns else 0
    spread_cov = df["market_spread"].notna().mean() if "market_spread" in df.columns else 0

    print(f"\nDone.")
    print(f"  Total games       : {len(df)}")
    print(f"  Moneyline cover   : {ml_cov:.1%}")
    print(f"  Total coverage    : {total_cov:.1%}")
    print(f"  Spread coverage   : {spread_cov:.1%}")

    return df


def _save_checkpoint(ml, totals, spreads, date_from, date_to):
    """Save in-progress scrape to checkpoint CSV."""
    if not ml:
        return
    slug = f"{date_from[:7]}_{date_to[:7]}".replace("-", "")
    path = f"data/odds/sbr_checkpoint_{slug}.csv"
    os.makedirs("data/odds", exist_ok=True)
    df = pd.DataFrame(ml)
    df.to_csv(path, index=False)


# ── Merge with model predictions ─────────────────────────────────────────────

def merge_with_predictions(preds_df: pd.DataFrame, lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join SBR lines onto model predictions DataFrame for backtesting.

    preds_df needs: GAME_DATE, HOME_TEAM_ABBREVIATION, AWAY_TEAM_ABBREVIATION
    lines_df needs: game_date, home_abbr, away_abbr, home_ml, away_ml
    """
    lines = lines_df.copy()
    lines["game_date"] = pd.to_datetime(lines["game_date"])

    preds = preds_df.copy()
    preds["GAME_DATE"] = pd.to_datetime(preds["GAME_DATE"])

    merged = preds.merge(
        lines[["game_date", "home_abbr", "away_abbr", "home_ml", "away_ml"]],
        left_on=["GAME_DATE", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"],
        right_on=["game_date", "home_abbr", "away_abbr"],
        how="left",
    )

    match_rate = merged["home_ml"].notna().mean()
    print(f"Lines matched: {match_rate:.1%} ({merged['home_ml'].notna().sum()} / {len(merged)} games)")
    return merged


# ── CLI ───────────────────────────────────────────────────────────────────────


def get_todays_moneylines() -> dict:
    """
    Scrape today's NBA moneylines from SBR.
    Returns dict keyed by (home_abbr, away_abbr) -> {"home": ml, "away": ml}
    Ready to pass directly to predict_today() as odds_dict.
    """
    from datetime import date
    import requests as _requests

    today = date.today().strftime("%Y-%m-%d")
    session = _requests.Session()

    games = fetch_moneylines(today, session)
    if not games:
        return {}

    odds_dict = {}
    for g in games:
        home = g.get("home_abbr", "")
        away = g.get("away_abbr", "")
        home_ml = g.get("home_ml")
        away_ml = g.get("away_ml")
        if home and away and home_ml and away_ml:
            odds_dict[(home.upper(), away.upper())] = {
                "home": int(home_ml),
                "away": int(away_ml),
            }

    return odds_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SBR NBA Moneyline + Totals + Spreads Scraper")
    parser.add_argument("--from",  dest="date_from", required=True,  help="Start date YYYY-MM-DD")
    parser.add_argument("--to",    dest="date_to",   required=True,  help="End date YYYY-MM-DD")
    parser.add_argument("--out",   default="data/odds/sbr_lines.csv", help="Output CSV path")
    parser.add_argument("--delay", type=float, default=2.0,           help="Seconds between dates (default 2.0)")
    parser.add_argument("--ml-only", action="store_true",             help="Only scrape moneylines (faster)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.ml_only:
        # Fast mode: moneyline only
        session = requests.Session()
        all_ml  = []
        for date in date_range(args.date_from, args.date_to):
            ml = fetch_moneylines(date, session)
            if ml:
                all_ml.extend(ml)
                print(f"  {date}: {len(ml)} games")
            else:
                print(f"  {date}: no games")
            time.sleep(args.delay)
        lines = pd.DataFrame(all_ml) if all_ml else pd.DataFrame()
    else:
        lines = scrape_range(args.date_from, args.date_to, delay=args.delay)

    if lines.empty:
        print("No data scraped.")
    else:
        lines.to_csv(args.out, index=False)
        print(f"\nSaved {len(lines)} games → {args.out}")
        print(f"Columns: {list(lines.columns)}")
        print(f"\nSample:")
        ml_cols = ["game_date", "home_abbr", "away_abbr", "home_ml", "away_ml"]
        available = [c for c in ml_cols if c in lines.columns]
        print(lines[available].head(5).to_string())