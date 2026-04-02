"""
scraper.py — NBA data scraper using the nba_api package.

Uses the nba_api library (pip install nba_api) which handles all the
headers, timeouts, and anti-scraping measures against stats.nba.com
automatically. Much more reliable than raw requests.

Caching / checkpoint system:
  - Each season is cached to data/cache/games_{season}.json
  - A checkpoint file records the last fetched game date per season
  - Re-runs only fetch games AFTER the checkpoint date (incremental)
  - Safe to run daily — never re-downloads what it already has
"""

import json
import time
import logging
import random
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats, scoreboardv2
from nba_api.stats.static import teams as nba_teams
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

import sys
_SRC_DIR  = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import (
    SEASONS, SEASON_TYPES, CACHE_DIR, RAW_DIR, REQUEST_DELAY
)

log = logging.getLogger("scraper")


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def _checkpoint_path(season: str, season_type: str) -> Path:
    slug = season.replace("-", "") + "_" + season_type.replace(" ", "")
    return CACHE_DIR / f"checkpoint_{slug}.json"


def load_checkpoint(season: str, season_type: str) -> dict:
    p = _checkpoint_path(season, season_type)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"last_date": None, "last_game_id": None, "games_fetched": 0}


def save_checkpoint(season: str, season_type: str, checkpoint: dict):
    p = _checkpoint_path(season, season_type)
    with open(p, "w") as f:
        json.dump(checkpoint, f, indent=2)
    log.debug(f"Checkpoint saved for {season} {season_type}: {checkpoint}")


def _cache_path(season: str, season_type: str) -> Path:
    slug = season.replace("-", "") + "_" + season_type.replace(" ", "")
    return CACHE_DIR / f"games_{slug}.json"


def load_cached_games(season: str, season_type: str) -> list[dict]:
    p = _cache_path(season, season_type)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def save_cached_games(season: str, season_type: str, games: list[dict]):
    p = _cache_path(season, season_type)
    with open(p, "w") as f:
        json.dump(games, f)
    log.info(f"  Cached {len(games)} total games -> {p.name}")


# ── Core fetch functions ─────────────────────────────────────────────────────

def fetch_season_game_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch all team game logs for a season using nba_api.
    Incremental — only fetches games after the last checkpoint date.
    """
    ckpt   = load_checkpoint(season, season_type)
    cached = load_cached_games(season, season_type)
    last_date = ckpt.get("last_date")

    log.info(f"Fetching {season} {season_type} | checkpoint: {last_date or 'none'} | {len(cached)} games cached")

    try:
        # Polite delay before hitting the API
        time.sleep(REQUEST_DELAY + random.uniform(0.5, 1.5))

        gamelog = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            league_id="00",
            date_from_nullable=last_date if last_date else "",
            date_to_nullable="",
            direction="ASC",
            sorter="DATE",
            player_or_team_abbreviation="T",
            timeout=60,
        )

        df = gamelog.get_data_frames()[0]

    except Exception as e:
        log.error(f"  nba_api fetch failed for {season}: {e}")
        log.error("  Tip: try again in a few minutes — stats.nba.com rate limits aggressively.")
        return pd.DataFrame(cached) if cached else pd.DataFrame()

    if df.empty:
        log.info(f"  No new games since {last_date}.")
        return pd.DataFrame(cached) if cached else pd.DataFrame()

    new_rows = df.to_dict(orient="records")

    # De-duplicate against existing cache
    cached_ids = {g["GAME_ID"] for g in cached}
    truly_new  = [r for r in new_rows if str(r["GAME_ID"]) not in cached_ids]

    if truly_new:
        log.info(f"  +{len(truly_new)} new game rows.")
        all_games = cached + truly_new
        save_cached_games(season, season_type, all_games)

        latest_date = max(str(r["GAME_DATE"]) for r in truly_new)
        latest_id   = max(str(r["GAME_ID"])   for r in truly_new)
        save_checkpoint(season, season_type, {
            "last_date":    latest_date,
            "last_game_id": latest_id,
            "games_fetched": len(all_games),
        })
    else:
        log.info(f"  No new games. Cache is up to date.")
        all_games = cached

    return pd.DataFrame(all_games)


def fetch_advanced_team_stats(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch season-level advanced team stats (pace, ORtg, DRtg etc.)
    Cached permanently per season.
    """
    cache_file = CACHE_DIR / f"advanced_{season.replace('-', '')}.json"

    if cache_file.exists():
        log.info(f"  Advanced stats for {season}: loaded from cache.")
        with open(cache_file) as f:
            return pd.DataFrame(json.load(f))

    try:
        time.sleep(REQUEST_DELAY + random.uniform(0.5, 1.0))
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=60,
        )
        df = stats.get_data_frames()[0]
        rows = df.to_dict(orient="records")
        with open(cache_file, "w") as f:
            json.dump(rows, f)
        log.info(f"  Advanced stats for {season}: fetched & cached ({len(rows)} teams).")
        return df
    except Exception as e:
        log.warning(f"  Advanced stats fetch failed for {season}: {e}")
        return pd.DataFrame()


def fetch_injury_report() -> pd.DataFrame:
    """
    Fetch current NBA injury report from ESPN public API.
    No auth required. Cached for 1 hour.

    Falls back to empty DataFrame if unavailable — injury data is
    supplementary and won't block the rest of the pipeline.
    """
    import requests
    cache_file = CACHE_DIR / "injury_report.json"
    CACHE_TTL  = 3600  # 1 hour

    # Return cached version if fresh enough
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL:
            log.info(f"  Injury report: using cache ({age/60:.0f}min old).")
            with open(cache_file) as f:
                return pd.DataFrame(json.load(f))

    # ESPN public injury API — no key, no auth, works globally
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    rows = []
    try:
        time.sleep(1.0)
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for team_entry in data.get("injuries", []):
            team_name = team_entry.get("team", {}).get("abbreviation", "")
            for injury in team_entry.get("injuries", []):
                athlete = injury.get("athlete", {})
                rows.append({
                    "team":   team_name,
                    "player": athlete.get("displayName", ""),
                    "status": injury.get("status", ""),
                    "reason": injury.get("longComment", injury.get("shortComment", "")),
                    "date":   injury.get("date", ""),
                })

        with open(cache_file, "w") as f:
            json.dump(rows, f)
        log.info(f"  Injury report: {len(rows)} players (ESPN).")

    except Exception as e:
        log.warning(f"  ESPN injury report failed: {e}")
        log.warning("  Injuries will be shown as unavailable. Predictions will still run.")

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["team", "player", "status", "reason", "date"]
    )


def fetch_schedule(season: str | None = None) -> pd.DataFrame:
    """
    Fetch full season schedule using nba_api.
    Cached — refreshed daily for current season.
    """
    if season is None:
        from src.predict import get_current_season
        season = get_current_season()
        log.info(f"  Schedule season auto-detected: {season}")
    cache_file = CACHE_DIR / f"schedule_{season.replace('-', '')}.json"

    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400:
            with open(cache_file) as f:
                return pd.DataFrame(json.load(f))

    try:
        time.sleep(REQUEST_DELAY)
        # Get today's scoreboard for upcoming games
        from nba_api.stats.endpoints import scheduleleaguev2
        sched = scheduleleaguev2.ScheduleLeagueV2(
            league_id="00",
            season=season,
        )
        df = sched.get_data_frames()[0]

        rows = []
        for _, row in df.iterrows():
            # Parse gameDate: "10/02/2025 00:00:00" -> "2025-10-02"
            raw_date = str(row.get("gameDate", "") or row.get("gameDateEst", ""))
            try:
                game_date = _pd.to_datetime(raw_date).strftime("%Y-%m-%d")
            except Exception:
                game_date = raw_date[:10] if raw_date else ""

            rows.append({
                "game_id":      str(row.get("gameId", "")),
                "game_date":    game_date,
                "home_team":    str(row.get("homeTeam_teamTricode", "")),
                "away_team":    str(row.get("awayTeam_teamTricode", "")),
                "home_team_id": str(row.get("homeTeam_teamId", "")),
                "away_team_id": str(row.get("awayTeam_teamId", "")),
                "status":       str(row.get("gameStatusText", "")),
            })

        with open(cache_file, "w") as f:
            json.dump(rows, f)
        log.info(f"  Schedule for {season}: {len(rows)} games.")
        return pd.DataFrame(rows)

    except Exception as e:
        log.warning(f"  Schedule fetch failed: {e}")
        return pd.DataFrame()


def fetch_all_seasons() -> pd.DataFrame:
    """
    Fetch and combine game logs for all configured seasons.
    Each season is fetched incrementally.
    """
    frames = []
    for season in SEASONS:
        for stype in SEASON_TYPES:
            df = fetch_season_game_log(season, stype)
            if not df.empty:
                df["SEASON"]      = season
                df["SEASON_TYPE"] = stype
                frames.append(df)

    if not frames:
        log.error("No data fetched!")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    out = RAW_DIR / "all_game_logs.parquet"
    combined.to_parquet(out, index=False)
    log.info(f"Raw data saved: {out} ({len(combined):,} rows)")
    return combined


def fetch_all_data(current_season: str | None = None) -> dict:
    """
    Master fetch function — pulls ALL datasets:
      1. Game logs for all seasons (incremental)
      2. Advanced team stats for all seasons (cached per season)
      3. Current season schedule (cached 24h)
      4. Live injury report (cached 1h)

    Returns a dict with keys: game_logs, advanced_stats, schedule, injuries
    """
    # Auto-detect current season if not provided
    if current_season is None:
        from src.predict import get_current_season
        current_season = get_current_season()

    results = {}

    # ── 1. Game logs (all seasons, incremental) ──────────────────────────
    log.info("")
    log.info("  [1/4] Game logs (all seasons)...")
    results["game_logs"] = fetch_all_seasons()
    n = len(results["game_logs"])
    log.info(f"  [1/4] Done - {n:,} team-game rows across {results['game_logs']['SEASON'].nunique() if n else 0} seasons")

    # ── 2. Advanced team stats (per season, permanent cache) ─────────────
    log.info("")
    log.info("  [2/4] Advanced team stats (per season)...")
    adv_frames = []
    for season in SEASONS:
        df = fetch_advanced_team_stats(season)
        if not df.empty:
            df["SEASON"] = season
            adv_frames.append(df)
        time.sleep(REQUEST_DELAY)

    if adv_frames:
        adv_combined = pd.concat(adv_frames, ignore_index=True)
        adv_out = RAW_DIR / "advanced_team_stats.parquet"
        adv_combined.to_parquet(adv_out, index=False)
        results["advanced_stats"] = adv_combined
        log.info(f"  [2/4] Done - {len(adv_combined):,} rows saved to advanced_team_stats.parquet")
    else:
        results["advanced_stats"] = pd.DataFrame()
        log.warning("  [2/4] No advanced stats fetched.")

    # ── 3. Schedule (current season, cached 24h) ─────────────────────────
    log.info("")
    log.info(f"  [3/4] Schedule for {current_season}...")
    results["schedule"] = fetch_schedule(current_season)
    n = len(results["schedule"])
    if not results["schedule"].empty:
        sched_slug = current_season.replace("-", "")
        sched_out  = RAW_DIR / f"schedule_{sched_slug}.parquet"
        results["schedule"].to_parquet(sched_out, index=False)
        log.info(f"  [3/4] Done - {n:,} games saved to schedule_{sched_slug}.parquet")
    else:
        log.warning(f"  [3/4] No schedule data found for {current_season}.")

    # ── 4. Injury report (live, cached 1h) ───────────────────────────────
    log.info("")
    log.info("  [4/4] Live injury report...")
    results["injuries"] = fetch_injury_report()
    n = len(results["injuries"])
    if not results["injuries"].empty:
        inj_out = RAW_DIR / "injury_report.parquet"
        results["injuries"].to_parquet(inj_out, index=False)
        log.info(f"  [4/4] Done - {n} players saved to {inj_out.name}")
    else:
        log.warning("  [4/4] No injury data saved.")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("")
    log.info("  ===== FETCH COMPLETE — FILES SAVED =====")
    log.info(f"  data/raw/all_game_logs.parquet          : {len(results['game_logs']):,} rows")
    log.info(f"  data/raw/advanced_team_stats.parquet    : {len(results['advanced_stats']):,} rows")
    log.info(f"  data/raw/schedule_{current_season.replace('-','')}.parquet  : {len(results['schedule']):,} games")
    log.info(f"  data/raw/injury_report.parquet          : {len(results['injuries']):,} players")
    log.info("  ========================================")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = fetch_all_seasons()
    print(df.shape)
    print(df.head())
