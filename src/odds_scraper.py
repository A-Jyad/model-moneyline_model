"""
odds_scraper.py — Free NBA moneyline odds scraper.

Two sources:
  1. The Odds API (free tier) — live + upcoming odds, 500 req/month free
     Sign up: https://the-odds-api.com (no credit card)
     Set env: ODDS_API_KEY=your_key

  2. Action Network unofficial API — live + upcoming odds, no key needed
     Fallback when Odds API key not set

Caching:
  - Live odds cached for 15 minutes (data/cache/odds_live.json)
  - Historical snapshots saved to data/cache/odds_history/ by date
  - Never re-fetches within the cache window

Usage:
    from src.odds_scraper import get_todays_odds, get_odds_dict
    odds = get_odds_dict()   # {game_id: {"home": american, "away": american}}
"""

import os
import json
import time
import logging
from datetime import date, datetime
from pathlib import Path

import requests
import pandas as pd

import sys
_SRC_DIR  = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import CACHE_DIR, REQUEST_DELAY, REQUEST_TIMEOUT

log = logging.getLogger("odds_scraper")

LIVE_CACHE_FILE    = CACHE_DIR / "odds_live.json"
HISTORY_CACHE_DIR  = CACHE_DIR / "odds_history"
HISTORY_CACHE_DIR.mkdir(exist_ok=True)
LIVE_CACHE_TTL     = 900   # 15 minutes

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Standard team name -> abbreviation for The Odds API
ODDS_API_TEAM_MAP = {
    "Atlanta Hawks":          "ATL", "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN", "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI", "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL", "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET", "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU", "Indiana Pacers":         "IND",
    "LA Clippers":            "LAC", "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL", "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA", "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK", "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL", "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC", "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR", "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}

# Action Network team ID -> abbreviation
ACTION_TEAM_MAP = {
    2:  "BOS", 3:  "BKN", 4:  "NYK", 5:  "PHI", 6:  "TOR",
    7:  "CHI", 8:  "CLE", 9:  "DET", 10: "IND", 11: "MIL",
    12: "ATL", 13: "CHA", 14: "MIA", 15: "ORL", 16: "WAS",
    17: "DEN", 18: "MIN", 19: "OKC", 20: "POR", 21: "UTA",
    22: "GSW", 23: "LAC", 24: "LAL", 25: "PHX", 26: "SAC",
    27: "DAL", 28: "HOU", 29: "MEM", 30: "NOP", 31: "SAS",
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_live_cache() -> list | None:
    if LIVE_CACHE_FILE.exists():
        age = time.time() - LIVE_CACHE_FILE.stat().st_mtime
        if age < LIVE_CACHE_TTL:
            with open(LIVE_CACHE_FILE) as f:
                data = json.load(f)
            log.info(f"Odds: using live cache ({age/60:.0f}min old, {len(data)} games)")
            return data
    return None


def _save_live_cache(data: list):
    with open(LIVE_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # Also save to daily history for CLV analysis
    today = date.today().isoformat()
    history_file = HISTORY_CACHE_DIR / f"odds_{today}.json"
    if not history_file.exists():
        with open(history_file, "w") as f:
            json.dump({"date": today, "fetched_at": datetime.now().isoformat(),
                       "games": data}, f, indent=2)
        log.info(f"Odds snapshot saved: {history_file.name}")


# ── Source 1: The Odds API ────────────────────────────────────────────────────

def fetch_odds_api() -> list[dict]:
    """
    Fetch today's NBA moneylines from The Odds API.
    Free tier: 500 req/month. Returns [] if no key set.
    """
    if not ODDS_API_KEY:
        return []

    url = f"{ODDS_API_BASE}/sports/basketball_nba/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "h2h",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,caesars,pinnacle",
    }
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info(f"Odds API: {remaining} requests remaining this month")

        resp.raise_for_status()
        data = resp.json()
        log.info(f"Odds API: {len(data)} games fetched")
        return data
    except requests.HTTPError as e:
        if resp.status_code == 401:
            log.error("Invalid ODDS_API_KEY")
        elif resp.status_code == 422:
            log.error("Odds API quota exhausted")
        else:
            log.error(f"Odds API error: {e}")
        return []
    except Exception as e:
        log.warning(f"Odds API failed: {e}")
        return []


def parse_odds_api(data: list[dict]) -> list[dict]:
    """Parse Odds API response into standard format."""
    games = []
    for game in data:
        home_full = game.get("home_team", "")
        away_full = game.get("away_team", "")
        home_abbr = ODDS_API_TEAM_MAP.get(home_full, home_full[:3].upper())
        away_abbr = ODDS_API_TEAM_MAP.get(away_full, away_full[:3].upper())
        commence  = game.get("commence_time", "")

        # Collect odds per bookmaker
        books = {}
        for book in game.get("bookmakers", []):
            key = book["key"]
            for market in book.get("markets", []):
                if market.get("key") == "h2h":
                    home_odds = away_odds = None
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home_full:
                            home_odds = outcome["price"]
                        elif outcome["name"] == away_full:
                            away_odds = outcome["price"]
                    if home_odds and away_odds:
                        books[key] = {"home": home_odds, "away": away_odds}

        if not books:
            continue

        # Best line (highest odds on each side)
        all_home = [b["home"] for b in books.values()]
        all_away = [b["away"] for b in books.values()]

        # Consensus (average)
        def avg_american(odds_list):
            decs = [(o/100+1 if o > 0 else 100/abs(o)+1) for o in odds_list]
            avg_dec = sum(decs) / len(decs)
            return round((avg_dec-1)*100 if avg_dec >= 2 else -100/(avg_dec-1))

        games.append({
            "home_team":       home_abbr,
            "away_team":       away_abbr,
            "home_team_full":  home_full,
            "away_team_full":  away_full,
            "commence_time":   commence,
            "source":          "odds_api",
            "bookmakers":      books,
            "consensus_home":  avg_american(all_home),
            "consensus_away":  avg_american(all_away),
            "best_home":       max(all_home),
            "best_away":       max(all_away),
            "sharp_home":      books.get("pinnacle", {}).get("home"),
            "sharp_away":      books.get("pinnacle", {}).get("away"),
        })

    return games


# ── Source 2: Action Network (no key needed) ──────────────────────────────────

def fetch_action_network() -> list[dict]:
    """
    Fetch today's NBA odds from Action Network's unofficial API.
    No API key required. Returns consensus moneylines.
    """
    today = date.today().isoformat()
    url   = f"https://api.actionnetwork.com/web/v1/games?league=nba&date={today}"
    headers = {
        "User-Agent":  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":      "application/json",
        "Referer":     "https://www.actionnetwork.com/",
        "Origin":      "https://www.actionnetwork.com",
    }
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        games_raw = data.get("games", [])
        log.info(f"Action Network: {len(games_raw)} games fetched")
        return games_raw
    except Exception as e:
        log.warning(f"Action Network failed: {e}")
        return []


def parse_action_network(data: list[dict]) -> list[dict]:
    """Parse Action Network response into standard format."""
    games = []
    for g in data:
        teams = g.get("teams", [])
        if len(teams) < 2:
            continue

        # Identify home/away
        home_team = away_team = None
        for t in teams:
            tid = t.get("id")
            abbr = ACTION_TEAM_MAP.get(tid, "???")
            if t.get("is_home"):
                home_team = abbr
            else:
                away_team = abbr

        if not home_team or not away_team:
            continue

        # Get consensus moneyline from odds object
        odds = g.get("odds", [{}])[0] if g.get("odds") else {}
        home_ml = odds.get("ml_home")
        away_ml = odds.get("ml_away")

        if not home_ml or not away_ml:
            continue

        games.append({
            "home_team":      home_team,
            "away_team":      away_team,
            "commence_time":  g.get("start_time", ""),
            "source":         "action_network",
            "bookmakers":     {"consensus": {"home": home_ml, "away": away_ml}},
            "consensus_home": int(home_ml),
            "consensus_away": int(away_ml),
            "best_home":      int(home_ml),
            "best_away":      int(away_ml),
            "sharp_home":     None,
            "sharp_away":     None,
        })

    return games


# ── Main entry point ──────────────────────────────────────────────────────────

def get_todays_odds(force_refresh: bool = False) -> list[dict]:
    """
    Get today's NBA moneylines from the best available source.
    Tries: cached -> SBR scraper -> Odds API (if key set) -> Action Network

    Returns list of game dicts with home/away American odds.
    """
    # Check cache first
    if not force_refresh:
        cached = _load_live_cache()
        if cached:
            return cached

    # Try SBR scraper first (free, no key, same source as backtest)
    log.info("Trying SBR scraper...")
    try:
        from src.sbr_scraper import get_todays_moneylines
        sbr_odds = get_todays_moneylines()
        if sbr_odds:
            # Convert to list format for caching
            games = [
                {
                    "home_team":      h,
                    "away_team":      a,
                    "source":         "sbr",
                    "consensus_home": v["home"],
                    "consensus_away": v["away"],
                    "best_home":      v["home"],
                    "best_away":      v["away"],
                    "sharp_home":     None,
                    "sharp_away":     None,
                }
                for (h, a), v in sbr_odds.items()
            ]
            _save_live_cache(games)
            log.info(f"Odds loaded from SBR: {len(games)} games")
            return games
        else:
            log.info("SBR returned no games for today")
    except Exception as e:
        log.warning(f"SBR scraper failed: {e}")

    # Try Odds API (needs key)
    if ODDS_API_KEY:
        raw = fetch_odds_api()
        if raw:
            games = parse_odds_api(raw)
            if games:
                _save_live_cache(games)
                log.info(f"Odds loaded from The Odds API: {len(games)} games")
                return games

    # Fallback: Action Network (no key, unofficial)
    log.info("Trying Action Network (no key required)...")
    raw = fetch_action_network()
    if raw:
        games = parse_action_network(raw)
        if games:
            _save_live_cache(games)
            log.info(f"Odds loaded from Action Network: {len(games)} games")
            return games

    log.warning("No odds available from any source.")
    log.warning("To get odds: set ODDS_API_KEY env var (free at https://the-odds-api.com)")
    return []


def get_odds_dict(schedule_df: pd.DataFrame | None = None,
                   force_refresh: bool = False) -> dict:
    """
    Main interface for predict.py and edge.py.

    Returns: {(home_abbr, away_abbr): {"home": american, "away": american}}
    Keyed by team pair so predict.py can look up by matchup.
    """
    games = get_todays_odds(force_refresh=force_refresh)

    odds_dict = {}
    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        # Use sharp (Pinnacle) if available, else consensus
        h = g.get("sharp_home") or g.get("consensus_home")
        a = g.get("sharp_away") or g.get("consensus_away")
        if h and a:
            odds_dict[(home.upper(), away.upper())] = {
                "home": int(h), "away": int(a)
            }

    log.info(f"Odds dict built: {len(odds_dict)} matchups")
    return odds_dict


def print_todays_lines():
    """Print a formatted odds board to console."""
    games = get_todays_odds()
    if not games:
        print("No odds available. Set ODDS_API_KEY or try later.")
        return

    print(f"\nNBA Odds — {date.today()}  ({games[0].get('source','?')})")
    print("─" * 55)
    print(f"  {'Matchup':30s} {'Home':8s} {'Away':8s}")
    print("─" * 55)
    for g in games:
        matchup = f"{g['away_team']} @ {g['home_team']}"
        h = g['consensus_home']
        a = g['consensus_away']
        h_str = f"+{h}" if h > 0 else str(h)
        a_str = f"+{a}" if a > 0 else str(a)
        print(f"  {matchup:30s} {h_str:8s} {a_str:8s}")
    print("─" * 55)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print_todays_lines()