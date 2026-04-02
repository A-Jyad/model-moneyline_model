import sys
from pathlib import Path

# Ensure project root is on sys.path however the script is invoked
_SRC_DIR  = Path(__file__).resolve().parent          # .../nba_predictor/src
_ROOT_DIR = _SRC_DIR.parent                          # .../nba_predictor
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
"""
elo.py — Standalone Elo rating system.

Used both inside feature engineering and for quick win probability estimates
without the full ML pipeline. Also exports current team ratings for live prediction.
"""

import json
import logging

import pandas as pd
import numpy as np


from config.settings import ELO_K, ELO_START, ELO_REGRESS_FRAC, CACHE_DIR, HOME_COURT_EDGE

log = logging.getLogger("elo")

ELO_STATE_FILE = CACHE_DIR / "elo_state.json"


class EloSystem:
    """
    Full Elo rating tracker with:
    - Season-to-season regression
    - Home court adjustment
    - Win probability output
    - State persistence (save/load)
    """

    def __init__(self, k: float = ELO_K, start: float = ELO_START,
                 regress_frac: float = ELO_REGRESS_FRAC,
                 home_advantage: float = HOME_COURT_EDGE):
        self.k = k
        self.start = start
        self.regress_frac = regress_frac
        self.home_adv = home_advantage  # points, converted to Elo below
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []
        self._current_season: str | None = None

        # Convert home court point advantage to Elo scale
        # ~3 points ≈ 45 Elo points at NBA pace
        self.home_elo_boost = home_advantage * 15

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.start)

    def _regress_season(self):
        """Regress all ratings toward mean at season start."""
        for t in self.ratings:
            self.ratings[t] = (
                self.ratings[t] * (1 - self.regress_frac)
                + self.start * self.regress_frac
            )

    def win_probability(self, home_team: str, away_team: str,
                        apply_home_boost: bool = True) -> float:
        """Return P(home win) given current ratings."""
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)
        boost = self.home_elo_boost if apply_home_boost else 0
        return 1 / (1 + 10 ** ((r_away - r_home - boost) / 400))

    def update(self, home_team: str, away_team: str,
               home_won: int, season: str,
               game_date: str = "", game_id: str = ""):
        """
        Update ratings after a game result.
        home_won: 1 if home team won, 0 if away won.
        """
        # Season transition
        if season != self._current_season:
            if self._current_season is not None:
                self._regress_season()
            self._current_season = season

        p_home = self.win_probability(home_team, away_team)
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)

        new_r_home = r_home + self.k * (home_won - p_home)
        new_r_away = r_away + self.k * ((1 - home_won) - (1 - p_home))

        self.history.append({
            "game_id":   game_id,
            "game_date": game_date,
            "season":    season,
            "home_team": home_team,
            "away_team": away_team,
            "home_won":  home_won,
            "p_home_pre": round(p_home, 4),
            "elo_home_pre": round(r_home, 1),
            "elo_away_pre": round(r_away, 1),
            "elo_home_post": round(new_r_home, 1),
            "elo_away_post": round(new_r_away, 1),
        })

        self.ratings[home_team] = new_r_home
        self.ratings[away_team] = new_r_away

    def fit(self, games_df: pd.DataFrame):
        """
        Fit Elo ratings on a chronological game DataFrame.
        Required columns: GAME_DATE, HOME_TEAM_ABBREVIATION, AWAY_TEAM_ABBREVIATION,
                          HOME_WIN, SEASON
        """
        df = games_df.sort_values("GAME_DATE").copy()

        home_col = next((c for c in ["HOME_TEAM_ABBREVIATION", "HOME_TEAM_ABBR"] if c in df.columns), None)
        away_col = next((c for c in ["AWAY_TEAM_ABBREVIATION", "AWAY_TEAM_ABBR"] if c in df.columns), None)

        if home_col is None or away_col is None:
            log.error("Elo.fit: missing team columns.")
            return self

        for _, row in df.iterrows():
            self.update(
                home_team = str(row[home_col]),
                away_team = str(row[away_col]),
                home_won  = int(row.get("HOME_WIN", 0)),
                season    = str(row.get("SEASON", "")),
                game_date = str(row.get("GAME_DATE", "")),
                game_id   = str(row.get("GAME_ID", "")),
            )

        log.info(f"Elo fit complete. {len(self.ratings)} teams rated.")
        return self

    def current_ratings(self) -> pd.DataFrame:
        """Return current Elo ratings as a sorted DataFrame."""
        return (
            pd.DataFrame(list(self.ratings.items()), columns=["team", "elo"])
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path = ELO_STATE_FILE):
        state = {
            "ratings": self.ratings,
            "current_season": self._current_season,
            "games_processed": len(self.history),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        log.info(f"Elo state saved: {path}")

    def load(self, path: Path = ELO_STATE_FILE):
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            self.ratings = state.get("ratings", {})
            self._current_season = state.get("current_season")
            log.info(f"Elo state loaded: {len(self.ratings)} teams, season={self._current_season}")
        else:
            log.info("No Elo state file found — starting fresh.")
        return self

    def predict_game(self, home_team: str, away_team: str) -> dict:
        """Quick prediction for a single game."""
        p = self.win_probability(home_team, away_team)
        return {
            "home_team":       home_team,
            "away_team":       away_team,
            "home_elo":        round(self.get_rating(home_team), 1),
            "away_elo":        round(self.get_rating(away_team), 1),
            "p_home_win":      round(p, 4),
            "p_away_win":      round(1 - p, 4),
            "home_spread_est": round((p - 0.5) * 14, 1),  # rough spread estimate
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    elo = EloSystem()
    # Quick test
    elo.ratings = {"OKC": 1650, "MEM": 1480}
    print(elo.predict_game("OKC", "MEM"))
    print(elo.predict_game("MEM", "OKC"))
