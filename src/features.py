import sys
from pathlib import Path

# Ensure project root is on sys.path however the script is invoked
_SRC_DIR  = Path(__file__).resolve().parent          # .../nba_predictor/src
_ROOT_DIR = _SRC_DIR.parent                          # .../nba_predictor
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
"""
features.py — Feature engineering pipeline.

Takes raw game log DataFrame and produces a game-level feature matrix
where each row = one game, from the HOME team's perspective.

Features built:
  - Rolling team offense/defense ratings (5/10/20 game windows, decay-weighted)
  - Elo ratings (updated game-by-game)
  - Rest advantage (days since last game, back-to-back flag)
  - Travel proxy (home/away streak)
  - Opponent-adjusted efficiency
  - Season SRS-style power rating
  - Win streak / momentum
"""

import logging

import numpy as np
import pandas as pd


from config.settings import (
    ROLLING_WINDOWS, DECAY_HALFLIFE, HOME_COURT_EDGE,
    BOX_COLS, ELO_K, ELO_START, ELO_REGRESS_FRAC,
    RAW_DIR, PROC_DIR,
)

log = logging.getLogger("features")



# ── Injury Features ───────────────────────────────────────────────────────────

# ESPN status -> impact score (probability player is missing the game)
INJURY_IMPACT = {
    "Out":          1.00,
    "Doubtful":     0.75,
    "Questionable": 0.50,
    "Day-To-Day":   0.25,
    "Probable":     0.10,
}


def build_injury_features(injury_df: pd.DataFrame) -> dict[str, float]:
    """
    Convert injury report into per-team impact scores.

    Returns a dict:
      {
        "ATL": {"players_out": 2, "impact_score": 1.75},
        "BOS": {"players_out": 0, "impact_score": 0.0},
        ...
      }

    Called during live prediction only. Historical training rows
    default to 0 (we cannot retroactively know injury status).
    """
    if injury_df is None or injury_df.empty:
        return {}

    team_impacts = {}
    for _, row in injury_df.iterrows():
        team   = str(row.get("team", "")).strip().upper()
        status = str(row.get("status", "")).strip()
        impact = INJURY_IMPACT.get(status, 0.0)

        if team not in team_impacts:
            team_impacts[team] = {"players_out": 0, "impact_score": 0.0}

        if status in ("Out", "Doubtful"):
            team_impacts[team]["players_out"] += 1

        team_impacts[team]["impact_score"] += impact

    return team_impacts


def get_injury_features_for_game(home_team: str, away_team: str,
                                  injury_df: pd.DataFrame | None) -> dict:
    """
    Return injury feature dict for a single game matchup.
    Safe to call with None injury_df — returns zeros.
    """
    zeros = {
        "home_players_out":   0,
        "away_players_out":   0,
        "home_injury_impact": 0.0,
        "away_injury_impact": 0.0,
        "injury_impact_diff": 0.0,
    }

    if injury_df is None or injury_df.empty:
        return zeros

    impacts = build_injury_features(injury_df)

    home_data = impacts.get(home_team, {"players_out": 0, "impact_score": 0.0})
    away_data = impacts.get(away_team, {"players_out": 0, "impact_score": 0.0})

    return {
        "home_players_out":   home_data["players_out"],
        "away_players_out":   away_data["players_out"],
        "home_injury_impact": round(home_data["impact_score"], 3),
        "away_injury_impact": round(away_data["impact_score"], 3),
        "injury_impact_diff": round(
            home_data["impact_score"] - away_data["impact_score"], 3
        ),
    }


# ── Data Cleaning ────────────────────────────────────────────────────────────

def clean_gamelogs(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize raw NBA Stats game log DataFrame."""
    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.upper()

    # Parse date
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    elif "GAME_DATE_EST" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE_EST"])

    # Extract home/away from MATCHUP column (e.g. "LAL vs. GSW" or "LAL @ GSW")
    if "MATCHUP" in df.columns:
        df["IS_HOME"] = df["MATCHUP"].str.contains(r" vs\. ", na=False).astype(int)
        df["OPP_ABBR"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s+(.+)$")[0].str.strip()
    else:
        df["IS_HOME"] = 0
        df["OPP_ABBR"] = ""

    # Ensure numeric
    num_cols = BOX_COLS + ["MIN", "FGA", "FTA", "FG3A", "OREB", "DREB"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing with 0
    df[BOX_COLS] = df[BOX_COLS].fillna(0)

    # Win flag
    if "WL" in df.columns:
        df["WIN"] = (df["WL"] == "W").astype(int)

    # Sort chronologically per team
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    log.info(f"Cleaned data: {df.shape[0]:,} rows, {df['TEAM_ABBREVIATION'].nunique()} teams")
    return df


# ── Rolling Features ─────────────────────────────────────────────────────────

def _exp_decay_mean(series: pd.Series, halflife: int) -> pd.Series:
    """Exponentially weighted mean with given halflife (games)."""
    return series.ewm(halflife=halflife, adjust=True).mean()


def build_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute rolling stats over multiple windows.
    Result is merged back — each game row gets the team's rolling stats
    computed BEFORE that game (shift by 1 to avoid lookahead).
    """
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).copy()
    feature_frames = []

    for team, grp in df.groupby("TEAM_ABBREVIATION"):
        grp = grp.copy().reset_index(drop=True)
        feats = {"GAME_ID": grp["GAME_ID"], "TEAM_ABBREVIATION": grp["TEAM_ABBREVIATION"]}

        # Rolling windows
        for w in ROLLING_WINDOWS:
            for col in BOX_COLS:
                if col in grp.columns:
                    rolled = grp[col].shift(1).rolling(w, min_periods=1).mean()
                    feats[f"{col}_roll{w}"] = rolled

        # Decay-weighted (captures recent form better)
        for col in BOX_COLS:
            if col in grp.columns:
                feats[f"{col}_ewm"] = _exp_decay_mean(grp[col].shift(1), DECAY_HALFLIFE)

        # Rest days since last game
        feats["days_rest"] = grp["GAME_DATE"].diff().dt.days.fillna(3)
        feats["is_b2b"] = (feats["days_rest"] == 1).astype(int)

        # Win streak (number of consecutive wins before this game)
        wins_shifted = grp["WIN"].shift(1).fillna(0)
        streak = []
        cur = 0
        for w_val in wins_shifted:
            if w_val == 1:
                cur += 1
            else:
                cur = 0
            streak.append(cur)
        feats["win_streak"] = streak

        # Home/away streak (proxy for travel fatigue)
        home_shifted = grp["IS_HOME"].shift(1).fillna(1)
        away_streak = []
        cur = 0
        for h in home_shifted:
            if h == 0:
                cur += 1
            else:
                cur = 0
            away_streak.append(cur)
        feats["away_streak"] = away_streak

        feature_frames.append(pd.DataFrame(feats))

    rolling_df = pd.concat(feature_frames, ignore_index=True)
    df = df.merge(rolling_df, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    log.info(f"Rolling features added: {df.shape[1]} columns total")
    return df


# ── Elo Rating ───────────────────────────────────────────────────────────────

def build_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute running Elo ratings.
    Returns df with ELO_PRE (rating before the game) for each row.
    Regresses ratings to mean at the start of each new season.
    """
    df = df.sort_values("GAME_DATE").copy()

    ratings: dict[str, float] = {}
    current_season = None
    elo_pre = []

    for _, row in df.iterrows():
        team = row["TEAM_ABBREVIATION"]
        season = row.get("SEASON", "")

        # Season regression to mean
        if season != current_season and current_season is not None:
            for t in ratings:
                ratings[t] = ratings[t] * (1 - ELO_REGRESS_FRAC) + ELO_START * ELO_REGRESS_FRAC
            current_season = season
        elif current_season is None:
            current_season = season

        r = ratings.get(team, ELO_START)
        elo_pre.append(r)

        # Update after game
        opp = row.get("OPP_ABBR", "")
        opp_r = ratings.get(opp, ELO_START)
        expected = 1 / (1 + 10 ** ((opp_r - r) / 400))
        actual = row.get("WIN", 0.5)
        ratings[team] = r + ELO_K * (actual - expected)

    df["ELO_PRE"] = elo_pre
    log.info("Elo ratings computed.")
    return df







# ── Advanced Derived Features ─────────────────────────────────────────────────

def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add eFG% (effective field goal %) rolling features.

    eFG% = (FGM + 0.5 * FG3M) / FGA
    Better than raw FG% because it accounts for the value of 3-pointers.
    In testing, HOME_eFG_PCT_roll10 has 0.078 correlation with HOME_WIN —
    one of the strongest single features in the model.

    Only eFG% is added here. Other candidates (TOV%, net rating, season
    progress, home win rate) showed correlation < 0.05 and added noise.
    """
    df = df.copy()
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])

    # Compute true eFG% only when raw box score columns are available
    # eFG% = (FGM + 0.5 * FG3M) / FGA
    # When not available, skip — approximation (FG_PCT * constant) adds no new info
    # since FG_PCT is already in the feature set
    if all(c in df.columns for c in ['FGM', 'FGA', 'FG3M']):
        df['eFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, np.nan)
        df['eFG_PCT'] = df['eFG_PCT'].fillna(df['FG_PCT'])
        df['eFG_PCT_roll10'] = (
            df.groupby('TEAM_ABBREVIATION')['eFG_PCT']
              .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )
        log.info("Advanced features added: true eFG% (FGM/FGA/FG3M available)")
    else:
        log.info("Skipping eFG%: FGM/FGA/FG3M not in raw data (approximation would duplicate FG_PCT)")

    return df

# ── Game-Level Feature Matrix ─────────────────────────────────────────────────

def build_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform team-game rows into one row per game (home vs away perspective).
    The target variable is HOME_WIN (1 if home team won).
    """
    # We need home and away rows paired on GAME_ID
    home = df[df["IS_HOME"] == 1].copy()
    away = df[df["IS_HOME"] == 0].copy()

    # Rename columns with prefixes
    home_cols = {c: f"HOME_{c}" for c in home.columns if c not in ["GAME_ID", "GAME_DATE", "SEASON"]}
    away_cols = {c: f"AWAY_{c}" for c in away.columns if c not in ["GAME_ID", "GAME_DATE", "SEASON"]}

    home = home.rename(columns=home_cols)
    away = away.rename(columns=away_cols)

    # Merge on game
    games = home.merge(
        away, on=["GAME_ID", "GAME_DATE", "SEASON"],
        suffixes=("", "_dup"), how="inner"
    )

    # Drop duplicate columns
    dup_cols = [c for c in games.columns if c.endswith("_dup")]
    games = games.drop(columns=dup_cols)

    # Target
    if "HOME_WIN" in games.columns:
        games["HOME_WIN"] = games["HOME_WIN"].astype(int)

    # Differential features (home minus away)
    DIFF_BASE_COLS = BOX_COLS + ["eFG_PCT"]
    for col in DIFF_BASE_COLS:
        for suffix in [f"_roll{w}" for w in ROLLING_WINDOWS] + ["_ewm", "_roll10"]:
            h_col = f"HOME_{col}{suffix}"
            a_col = f"AWAY_{col}{suffix}"
            if h_col in games.columns and a_col in games.columns:
                games[f"DIFF_{col}{suffix}"] = games[h_col] - games[a_col]

    # Elo differential
    if "HOME_ELO_PRE" in games.columns and "AWAY_ELO_PRE" in games.columns:
        games["ELO_DIFF"] = games["HOME_ELO_PRE"] - games["AWAY_ELO_PRE"]

    # Rest advantage
    if "HOME_days_rest" in games.columns and "AWAY_days_rest" in games.columns:
        games["REST_DIFF"] = games["HOME_days_rest"] - games["AWAY_days_rest"]
        games["HOME_IS_B2B"] = games.get("HOME_is_b2b", 0)
        games["AWAY_IS_B2B"] = games.get("AWAY_is_b2b", 0)

    # Win streak differential
    if "HOME_win_streak" in games.columns and "AWAY_win_streak" in games.columns:
        games["STREAK_DIFF"] = games["HOME_win_streak"] - games["AWAY_win_streak"]

    log.info(f"Game feature matrix: {len(games):,} games, {games.shape[1]} columns")
    return games.sort_values("GAME_DATE").reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (exclude metadata & target)."""
    exclude = {
        # Identifiers / metadata
        "GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE",
        "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION",
        "HOME_TEAM_NAME", "AWAY_TEAM_NAME",
        "HOME_TEAM_ID", "AWAY_TEAM_ID",
        "HOME_SEASON_ID", "AWAY_SEASON_ID",
        "HOME_MATCHUP", "AWAY_MATCHUP",
        "HOME_OPP_ABBR", "AWAY_OPP_ABBR",
        "HOME_VIDEO_AVAILABLE", "AWAY_VIDEO_AVAILABLE",
        "HOME_GAME_DATE", "AWAY_GAME_DATE",
        "HOME_SEASON_TYPE", "AWAY_SEASON_TYPE",
        "HOME_SEASON", "AWAY_SEASON",
        "HOME_IS_HOME", "AWAY_IS_HOME",
        # ── TARGET AND ITS EQUIVALENTS — must all be excluded ──
        "HOME_WIN",   # the target variable
        "AWAY_WIN",   # inverse of HOME_WIN — perfect leakage
        "HOME_WL",    # string version of HOME_WIN
        "AWAY_WL",    # string version of AWAY_WIN
        "HOME_PLUS_MINUS",  # raw game result (points diff)
        "AWAY_PLUS_MINUS",  # raw game result
    }
    # Remove ALL raw game-result stats — anything that describes what happened
    # during the game itself (not pre-game rolling averages).
    # This covers both BOX_COLS and the extra columns the real NBA API returns.
    ALL_GAME_RESULT_COLS = [
        # Core box score (BOX_COLS)
        "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
        "REB", "AST", "TOV", "STL", "BLK", "PLUS_MINUS",
        # Additional NBA API columns not in BOX_COLS
        "FGM", "FGA",           # field goals made/attempted
        "FG3M", "FG3A",         # 3-pointers made/attempted
        "FTM", "FTA",           # free throws made/attempted
        "OREB", "DREB",         # offensive/defensive rebounds
        "PF",                   # personal fouls
        "MIN",                  # minutes (always 240, no signal)
        # Season/team identifiers that should not be features
        "SEASON_ID", "TEAM_ID", "TEAM_NAME",
        "VIDEO_AVAILABLE", "IS_HOME",
        # Raw advanced stats (use rolled version only)
        "eFG_PCT",
    ]
    raw_box = set()
    for col in ALL_GAME_RESULT_COLS:
        raw_box.add(f"HOME_{col}")
        raw_box.add(f"AWAY_{col}")

    feat_cols = []
    for c in df.columns:
        if c in exclude or c in raw_box:
            continue
        # Must be numeric and not datetime/object
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        # Skip raw home/away PTS to prevent lookahead
        if c in ("HOME_PTS", "AWAY_PTS"):
            continue
        feat_cols.append(c)

    return feat_cols


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_feature_pipeline(raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
      raw game logs → cleaned → rolling features → Elo → game matrix
    """
    if raw_df is None:
        raw_path = RAW_DIR / "all_game_logs.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {raw_path}. Run --fetch first.")
        raw_df = pd.read_parquet(raw_path)
        log.info(f"Loaded raw data: {raw_df.shape}")

    df = clean_gamelogs(raw_df)
    df = build_team_rolling_features(df)
    df = build_elo_ratings(df)
    df = build_advanced_features(df)
    game_df = build_game_features(df)

    out_path = PROC_DIR / "game_features.parquet"
    game_df.to_parquet(out_path, index=False)
    log.info(f"Feature matrix saved: {out_path}")
    return game_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run_feature_pipeline()
    print(df.shape)
    print(df.dtypes.value_counts())