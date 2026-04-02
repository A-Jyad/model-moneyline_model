import sys
from pathlib import Path

# Ensure project root is on sys.path however the script is invoked
_SRC_DIR  = Path(__file__).resolve().parent          # .../nba_predictor/src
_ROOT_DIR = _SRC_DIR.parent                          # .../nba_predictor
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
"""
model.py — Ensemble model: Logistic Regression + XGBoost + LightGBM + Elo.

Training flow:
  1. Load feature matrix from data/processed/
  2. Split: train (all seasons except last 2), valid (2nd to last), test (last)
  3. Fit LR, XGBoost, LightGBM on train
  4. Calibrate each model on validation set (Platt scaling)
  5. Blend calibrated probabilities using configured weights
  6. Save all models + scaler to data/models/
"""

import logging
import joblib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, log_loss, accuracy_score
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb


from config.settings import (
    PROC_DIR, MODEL_DIR, RANDOM_SEED,
    LR_PARAMS, XGB_PARAMS, LGB_PARAMS, ENSEMBLE_WEIGHTS,
    TEST_SEASON, VALID_SEASON, SEASONS,
)
from src.features import get_feature_columns
from src.elo import EloSystem

log = logging.getLogger("model")





# ── Data Loading ──────────────────────────────────────────────────────────────

def load_feature_matrix() -> pd.DataFrame:
    path = PROC_DIR / "game_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {path}. Run --features first.")
    df = pd.read_parquet(path)
    log.info(f"Loaded feature matrix: {df.shape}")
    return df


def split_data(df: pd.DataFrame, feat_cols: list[str]):
    """
    Time-based split to prevent data leakage.
    Returns (X_train, y_train, X_valid, y_valid, X_test, y_test, meta_test)
    """
    # Exclude BOTH test seasons from training
    try:
        from config.settings import TEST_SEASON_2
        held_out = [VALID_SEASON, TEST_SEASON, TEST_SEASON_2]
    except ImportError:
        held_out = [VALID_SEASON, TEST_SEASON]

    train_mask = ~df["SEASON"].isin(held_out)
    valid_mask = df["SEASON"] == VALID_SEASON
    test_mask  = df["SEASON"] == TEST_SEASON

    def get_xy(mask):
        sub = df[mask].dropna(subset=feat_cols + ["HOME_WIN"])
        return sub[feat_cols].values, sub["HOME_WIN"].values, sub

    X_tr, y_tr, _    = get_xy(train_mask)
    X_va, y_va, _    = get_xy(valid_mask)
    X_te, y_te, meta = get_xy(test_mask)

    log.info(f"Train: {X_tr.shape[0]:,} | Valid: {X_va.shape[0]:,} | Test: {X_te.shape[0]:,}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, meta


# ── Individual Models ─────────────────────────────────────────────────────────

def _calibrate(estimator, X_valid, y_valid, method="sigmoid"):
    """
    Platt / isotonic calibration on a pre-fitted estimator.
    Works across sklearn versions by using cv=None + is_fitted pattern.
    """
    from sklearn.calibration import _CalibratedClassifier
    try:
        # sklearn >= 1.2 uses 'prefit' keyword
        cal = CalibratedClassifierCV(estimator, cv="prefit", method=method)
        cal.fit(X_valid, y_valid)
    except Exception:
        # Fallback: wrap raw probabilities with isotonic regression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression as LR

        raw_p = estimator.predict_proba(X_valid)[:, 1]
        if method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(raw_p.reshape(-1, 1), y_valid)
            cal = _PlattWrapper(estimator, ir)
        else:
            cal = estimator  # use as-is
    return cal


class _PlattWrapper:
    """Thin wrapper for calibrated predict_proba when sklearn API differs."""
    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw.reshape(-1, 1))
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_logistic(X_train, y_train, X_valid, y_valid):
    log.info("Training Logistic Regression…")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**LR_PARAMS)),
    ])
    pipe.fit(X_train, y_train)
    cal = _calibrate(pipe, X_valid, y_valid, method="sigmoid")
    p = cal.predict_proba(X_valid)[:, 1]
    log.info(f"  LR  — Brier: {brier_score_loss(y_valid, p):.4f} | AUC: {roc_auc_score(y_valid, p):.4f}")
    return cal


def train_xgboost(X_train, y_train, X_valid, y_valid):
    log.info("Training XGBoost…")
    params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    cal = _calibrate(model, X_valid, y_valid, method="isotonic")
    p = cal.predict_proba(X_valid)[:, 1]
    log.info(f"  XGB — Brier: {brier_score_loss(y_valid, p):.4f} | AUC: {roc_auc_score(y_valid, p):.4f}")
    return cal


def train_lightgbm(X_train, y_train, X_valid, y_valid):
    log.info("Training LightGBM…")
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    cal = _calibrate(model, X_valid, y_valid, method="isotonic")
    p = cal.predict_proba(X_valid)[:, 1]
    log.info(f"  LGB — Brier: {brier_score_loss(y_valid, p):.4f} | AUC: {roc_auc_score(y_valid, p):.4f}")
    return cal


# ── Ensemble ──────────────────────────────────────────────────────────────────

class NBAEnsemble:
    """
    Ensemble of LR + XGB + LGB + Elo.
    Blends calibrated probabilities with configured weights.
    """

    def __init__(self):
        self.lr  = None
        self.xgb = None
        self.lgb = None
        self.elo = EloSystem()
        self.feat_cols: list[str] = []
        self.weights = ENSEMBLE_WEIGHTS

    def fit(self, df: pd.DataFrame):
        self.feat_cols = get_feature_columns(df)
        X_tr, y_tr, X_va, y_va, X_te, y_te, meta_te = split_data(df, self.feat_cols)

        self.lr  = train_logistic(X_tr, y_tr, X_va, y_va)
        self.xgb = train_xgboost(X_tr, y_tr, X_va, y_va)
        self.lgb = train_lightgbm(X_tr, y_tr, X_va, y_va)

        # Fit Elo on all pre-test data
        pre_test = df[df["SEASON"] != TEST_SEASON].copy()
        self.elo.fit(pre_test)
        self.elo.save()

        log.info("Ensemble fit complete.")
        return self

    def predict_proba_components(self, X: np.ndarray,
                                  elo_probs: np.ndarray | None = None) -> dict:
        """Return dict of probability arrays, one per model."""
        import warnings
        X = np.asarray(X, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {
                "lr":  self.lr.predict_proba(X)[:, 1],
                "xgb": self.xgb.predict_proba(X)[:, 1],
                "lgb": self.lgb.predict_proba(X)[:, 1],
                "elo": elo_probs if elo_probs is not None else np.full(len(X), 0.5),
            }

    def blend(self, components: dict) -> np.ndarray:
        """Weighted blend of model probabilities."""
        w = self.weights
        total_w = sum(w[k] for k in components)
        return sum(components[k] * w[k] for k in components) / total_w

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict on a DataFrame with full metadata output."""
        feat_cols = [c for c in self.feat_cols if c in df.columns]
        X = df[feat_cols].fillna(0).values

        # Get Elo probs from pre-computed column if available
        if "ELO_DIFF" in df.columns:
            # Convert Elo diff to probability
            elo_probs = 1 / (1 + 10 ** (-df["ELO_DIFF"].values / 400))
        else:
            elo_probs = np.full(len(df), 0.5)

        comps = self.predict_proba_components(X, elo_probs)
        blended = self.blend(comps)

        out = df[["GAME_ID", "GAME_DATE", "SEASON",
                   "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"]].copy()
        out["P_HOME_WIN"]   = blended
        out["P_HOME_LR"]    = comps["lr"]
        out["P_HOME_XGB"]   = comps["xgb"]
        out["P_HOME_LGB"]   = comps["lgb"]
        out["P_HOME_ELO"]   = comps["elo"]
        if "HOME_WIN" in df.columns:
            out["HOME_WIN"] = df["HOME_WIN"].values
        return out

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate on a dataset. Returns dict of metrics."""
        preds = self.predict_df(df)
        preds = preds.dropna(subset=["HOME_WIN", "P_HOME_WIN"])
        y     = preds["HOME_WIN"].values
        p     = preds["P_HOME_WIN"].values

        acc   = accuracy_score(y, (p >= 0.5).astype(int))
        brier = brier_score_loss(y, p)
        auc   = roc_auc_score(y, p)
        ll    = log_loss(y, p)

        metrics = {
            "n_games": len(y),
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
            "roc_auc": round(auc, 4),
            "log_loss": round(ll, 4),
        }
        log.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.lr,        MODEL_DIR / "model_lr.pkl")
        joblib.dump(self.xgb,       MODEL_DIR / "model_xgb.pkl")
        joblib.dump(self.lgb,       MODEL_DIR / "model_lgb.pkl")
        joblib.dump(self.feat_cols, MODEL_DIR / "feat_cols.pkl")
        joblib.dump(self.weights,   MODEL_DIR / "weights.pkl")
        log.info(f"Models saved to {MODEL_DIR}")

    def load(self):
        self.lr        = joblib.load(MODEL_DIR / "model_lr.pkl")
        self.xgb       = joblib.load(MODEL_DIR / "model_xgb.pkl")
        self.lgb       = joblib.load(MODEL_DIR / "model_lgb.pkl")
        self.feat_cols = joblib.load(MODEL_DIR / "feat_cols.pkl")
        self.weights   = joblib.load(MODEL_DIR / "weights.pkl")
        self.elo.load()
        log.info("Models loaded.")
        return self

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance from XGBoost."""
        imp = self.xgb.estimator.feature_importances_  # type: ignore
        return (
            pd.DataFrame({"feature": self.feat_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def train_and_save() -> NBAEnsemble:
    df = load_feature_matrix()
    model = NBAEnsemble()
    model.fit(df)

    # Evaluate on test set
    test_df = df[df["SEASON"] == TEST_SEASON].copy()
    if len(test_df) > 0:
        metrics = model.evaluate(test_df)
        print("\n=== Test Set Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k:20s}: {v}")

    model.save()
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_and_save()