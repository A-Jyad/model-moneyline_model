"""
dashboard_app.py — NBA Model Dashboard
Run: python -m streamlit run dashboard_app.py
"""

import sys, json
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="NBA Model Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Ensure logs directory exists (needed in cloud environment)
(ROOT / "logs").mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def american_to_decimal(ml):
    try:
        ml = float(ml)
        return ml / 100 + 1 if ml > 0 else 100 / abs(ml) + 1
    except:
        return 1.909

def fmt_ml(ml):
    try:
        ml = int(float(ml))
        return f"+{ml}" if ml > 0 else str(ml)
    except:
        return str(ml)

# ── Storage ───────────────────────────────────────────────────────────────────

TRACKER_FILE = ROOT / "logs" / "bet_tracker.json"
TRACKER_FILE.parent.mkdir(exist_ok=True)

def load_bets():
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE) as f:
            return json.load(f)
    return []

def save_bets(bets):
    with open(TRACKER_FILE, "w") as f:
        json.dump(bets, f, indent=2, default=str)

# ── Backtest data ─────────────────────────────────────────────────────────────

LOG_DIR = ROOT / "logs"

BACKTEST_FILES = {
    "2023-24": LOG_DIR / "backtest_real_2024.csv",
    "2024-25": LOG_DIR / "backtest_real_2025.csv",
    "2025-26": LOG_DIR / "backtest_real_2026.csv",
}

CLEAN_SEASONS = ["2024-25", "2025-26"]

@st.cache_data(ttl=3600)
def load_all_backtest() -> pd.DataFrame:
    dfs = []
    for season, path in BACKTEST_FILES.items():
        if path.exists():
            df = pd.read_csv(path)
            df["season"] = season

            # Normalise column names — two CSV formats exist:
            # Format A (backtest_real_*.csv):  moneyline_home/away, home_won, pnl_per_unit
            # Format B (backtest_202425.csv):  home_odds/away_odds, home_actually_won, result_correct
            if "moneyline_home" in df.columns and "home_odds" not in df.columns:
                df = df.rename(columns={
                    "moneyline_home": "home_odds",
                    "moneyline_away": "away_odds",
                })
            if "home_won" not in df.columns and "home_actually_won" in df.columns:
                df["home_won"] = df["home_actually_won"]
            if "pnl_per_unit" not in df.columns and "pnl" in df.columns:
                df["pnl_per_unit"] = df["pnl"]
            if "result_correct" not in df.columns and "home_won" in df.columns:
                # result_correct = did the flagged bet win?
                pass  # will be computed in apply_filters from home_won

            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    for col in ["model_prob_home", "home_odds", "away_odds",
                "edge_home_pct", "edge_away_pct", "home_won",
                "result_correct", "pnl_per_unit", "bet_odds", "bet_edge_pct"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined

@st.cache_data(ttl=300)
def load_todays_predictions():
    today = date.today().strftime("%Y-%m-%d")
    f = LOG_DIR / f"predictions_{today}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

# ── Core filter engine ────────────────────────────────────────────────────────

def implied_prob(ml: float) -> float:
    """Convert American odds to implied probability (no vig removal)."""
    ml = float(ml)
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)


def remove_vig(p_home_raw: float, p_away_raw: float):
    """Remove vig to get fair probabilities."""
    total = p_home_raw + p_away_raw
    return p_home_raw / total, p_away_raw / total


def apply_filters(df: pd.DataFrame,
                  min_edge: float, max_edge: float,
                  min_odds: int, max_odds: int,
                  underdogs_only: bool,
                  seasons: list[str]) -> pd.DataFrame:
    """
    Re-evaluate which bets qualify under given filter settings.
    Computes edge live from model_prob_home + moneyline_home/away
    so any filter combination works in real time.
    """
    if df.empty:
        return pd.DataFrame()

    rows = df[df["season"].isin(seasons)].copy() if seasons else df.copy()
    if rows.empty:
        return pd.DataFrame()

    # Normalise column names once before the loop
    if "home_odds" not in rows.columns and "moneyline_home" in rows.columns:
        rows = rows.rename(columns={"moneyline_home": "home_odds",
                                    "moneyline_away": "away_odds"})
    if "home_won" not in rows.columns and "home_actually_won" in rows.columns:
        rows = rows.rename(columns={"home_actually_won": "home_won"})

    results = []
    for _, row in rows.iterrows():
        # Get raw values
        p_home = row.get("model_prob_home")
        ho     = row.get("home_odds")
        ao     = row.get("away_odds")
        hw     = row.get("home_won")

        # Skip if missing key data
        try:
            p_home = float(p_home)
            ho     = float(ho)
            ao     = float(ao)
        except (TypeError, ValueError):
            continue
        if pd.isna(p_home) or pd.isna(ho) or pd.isna(ao):
            continue

        p_away = 1.0 - p_home

        # Compute fair market probabilities (vig-removed)
        raw_h = implied_prob(ho)
        raw_a = implied_prob(ao)
        fair_h, fair_a = remove_vig(raw_h, raw_a)

        # Compute edge for each side
        eh = (p_home - fair_h) * 100
        ea = (p_away - fair_a) * 100

        # Pick best side
        if eh >= ea and eh >= min_edge:
            bet_odds, bet_edge, is_home = ho, eh, True
        elif ea >= min_edge:
            bet_odds, bet_edge, is_home = ao, ea, False
        else:
            continue

        # Apply filters
        if underdogs_only and bet_odds < 0:    continue  # block favourites
        if bet_odds > max_odds:                continue  # block huge longshots
        # min_odds only applies to underdogs (positive odds) — skip near-even dogs
        if bet_odds > 0 and bet_odds <= abs(min_odds): continue
        if bet_edge > max_edge:                continue  # block overconfident picks
        if pd.isna(hw):                        continue

        won = (is_home == (float(hw) == 1))
        dec = american_to_decimal(bet_odds)
        pnl = (dec - 1) if won else -1.0

        results.append({
            "season":   row["season"],
            "is_home":  is_home,
            "bet_odds": bet_odds,
            "bet_edge": round(bet_edge, 2),
            "won":      won,
            "pnl":      pnl,
        })

    return pd.DataFrame(results)


def summarise(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"bets": 0, "roi": 0, "wr": 0, "pnl": 0, "be": 0}
    wins = df["won"].sum()
    total = len(df)
    pnl  = df["pnl"].sum()
    avg_dec = df["bet_odds"].apply(american_to_decimal).mean()
    return {
        "bets": total,
        "roi":  round(pnl / total * 100, 2),
        "wr":   round(wins / total * 100, 1),
        "pnl":  round(pnl, 2),
        "be":   round(1 / avg_dec * 100, 1),
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🏀 NBA Model")
st.sidebar.caption(f"{date.today().strftime('%b %d, %Y')}  |  2025-26 Season")

page = st.sidebar.radio("", [
    "🏀 Today's Predictions",
    "🔬 Filter Playground",
    "📋 Bet Tracker",
    "📈 Performance",
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: TODAY'S PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏀 Today's Predictions":
    st.title("🏀 Today's Predictions")
    st.caption(date.today().strftime("%A, %B %d, %Y"))

    col_r, col_b = st.columns([4,1])
    with col_b:
        if st.button("🔄 Refresh Predictions", use_container_width=True):
            st.cache_data.clear()
            with st.spinner("Fetching latest games and odds..."):
                try:
                    from src.predict import predict_today, get_current_season
                    season = get_current_season()
                    preds  = predict_today(season=season)
                    if preds is not None and not preds.empty:
                        st.success(f"✅ Updated! {len(preds)} games loaded.")
                    else:
                        st.warning("No games found for today.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Try running `python predict.py` locally and pushing the predictions CSV.")

    preds = load_todays_predictions()

    if preds.empty:
        st.info("No predictions yet. Click Refresh or run: `python predict.py`")

        with st.expander("📋 Enter odds manually"):
            st.code('python predict.py --odds "HOME1:ML,AWAY1:ML;HOME2:ML,AWAY2:ML"')
            odds_input = st.text_input("Paste odds string:", placeholder='LAC:-180,SAS:+155;GSW:+220,CLE:-270')
            if st.button("Run") and odds_input:
                try:
                    sys.path.insert(0, str(ROOT))
                    from predict import parse_odds
                    from src.predict import predict_today, get_current_season
                    preds = predict_today(season=get_current_season(), odds_dict=parse_odds(odds_input))
                    preds.to_csv(LOG_DIR / f"predictions_{date.today()}.csv", index=False)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        # Detect flagged bets
        if "has_edge" in preds.columns:
            flagged = preds[preds["has_edge"] == True]
        else:
            flagged = preds[preds["recommendation"].astype(str).str.contains("BET", na=False)]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Games", len(preds))
        m2.metric("Flagged Bets", len(flagged),
                  delta=f"⭐ BET" if len(flagged) > 0 else None)
        m3.metric("Min Edge", "15%")
        m4.metric("Odds Range", "+141–+500")

        st.divider()

        # Flagged bets
        if len(flagged) > 0:
            st.subheader("⭐ Flagged Bets")
            for _, row in flagged.iterrows():
                rec = str(row.get("recommendation",""))
                bet_team = rec.split("BET ")[1].split(" ")[0] if "BET " in rec else "?"
                odds_str = rec.split("(")[1].split(")")[0] if "(" in rec else "?"
                edge_str = rec.split("Edge: ")[1].split("%")[0] if "Edge:" in rec else "?"
                ev_str   = rec.split("EV: ")[1] if "EV:" in rec else "—"
                p_home   = float(row.get("p_home_win", row.get("model_prob_home", 0.5)))
                kelly    = float(row.get("kelly_units", 0) or 0)

                with st.container():
                    c1, c2, c3, c4 = st.columns([3,2,2,2])
                    with c1:
                        st.markdown(f"### {row.get('home_team','?')} vs {row.get('away_team','?')}")
                        st.markdown(f"**🎯 BET {bet_team} ({odds_str})**")
                    c2.metric("Edge", f"{edge_str}%")
                    c3.metric("EV", ev_str)
                    c4.metric("Kelly %", f"{kelly:.1f}%")

                    if st.button(f"✅ Log bet — {bet_team}",
                                 key=f"log_{row.get('home_team','')}_{row.get('away_team','')}"):
                        bets = load_bets()
                        bets.append({
                            "date": date.today().isoformat(),
                            "home_team": row.get("home_team",""),
                            "away_team": row.get("away_team",""),
                            "bet_team":  bet_team,
                            "bet_odds":  odds_str,
                            "edge_pct":  edge_str,
                            "result":    "pending",
                            "units":     1.0,
                            "pnl":       None,
                        })
                        save_bets(bets)
                        st.success("Logged! Update result in Bet Tracker.")
                        st.rerun()
                    st.divider()

        # All games
        st.subheader("All Games")
        for _, row in preds.iterrows():
            home   = row.get("home_team","?")
            away   = row.get("away_team","?")
            p_home = float(row.get("p_home_win", row.get("model_prob_home", 0.5)))
            rec    = str(row.get("recommendation",""))
            is_bet = "BET" in rec and "NO BET" not in rec
            elo    = float(row.get("elo_diff", 0) or 0)
            b2b_h  = " [B2B]" if row.get("home_b2b") else ""
            b2b_a  = " [B2B]" if row.get("away_b2b") else ""

            # Extract odds and edge from recommendation or columns
            edge_h = row.get("edge_home_pct")
            edge_a = row.get("edge_away_pct")
            home_ml = row.get("home_ml", row.get("home_odds"))
            away_ml = row.get("away_ml", row.get("away_odds"))

            # Try to parse odds from recommendation if not in columns
            if home_ml is None and "(" in rec:
                try:
                    odds_part = rec.split("(")[1].split(")")[0]
                    # odds_part like "+142" or "-150"
                except:
                    pass

            c1, c2, c3 = st.columns([2, 4, 3])
            with c1:
                st.markdown(f"**{home}{b2b_h}** vs {away}{b2b_a}")
                st.caption(f"Elo: {elo:+.0f}")
            with c2:
                bar = "█" * int(p_home*20) + "░" * (20-int(p_home*20))
                st.markdown(f"`{bar}` {p_home*100:.0f}% / {(1-p_home)*100:.0f}%")
                if is_bet:
                    st.markdown(f"<span style='color:#3ddc84;font-weight:600'>⭐ {rec}</span>",
                                unsafe_allow_html=True)
                elif "NO BET" in rec:
                    st.markdown(f"<span style='color:#ff4b4b'>✗ {rec}</span>",
                                unsafe_allow_html=True)
                else:
                    st.caption(rec)
            with c3:
                # Show odds and edge for each side
                try:
                    if home_ml is not None and away_ml is not None:
                        hml = int(float(home_ml))
                        aml = int(float(away_ml))
                        h_str = f"+{hml}" if hml > 0 else str(hml)
                        a_str = f"+{aml}" if aml > 0 else str(aml)
                        eh = float(edge_h) if edge_h is not None and str(edge_h) != "nan" else None
                        ea = float(edge_a) if edge_a is not None and str(edge_a) != "nan" else None
                        if eh is not None:
                            h_color = "#3ddc84" if eh > 0 else "#ff4b4b"
                            a_color = "#3ddc84" if ea > 0 else "#ff4b4b"
                            st.markdown(
                                f"<div style='font-size:13px;line-height:1.8'>"
                                f"<span style='color:#aaa'>{home}:</span> "
                                f"<b>{h_str}</b> "
                                f"<span style='color:{h_color}'>({eh:+.1f}%)</span><br>"
                                f"<span style='color:#aaa'>{away}:</span> "
                                f"<b>{a_str}</b> "
                                f"<span style='color:{a_color}'>({ea:+.1f}%)</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"<div style='font-size:13px;line-height:1.8'>"
                                f"<span style='color:#aaa'>{home}:</span> <b>{h_str}</b><br>"
                                f"<span style='color:#aaa'>{away}:</span> <b>{a_str}</b>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.caption("No odds")
                except Exception:
                    st.caption("No odds")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: FILTER PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Filter Playground":
    st.title("🔬 Filter Playground")
    st.caption("Adjust filters and instantly see the impact on backtest performance across all seasons.")

    raw = load_all_backtest()
    available_seasons = sorted(raw["season"].unique().tolist()) if not raw.empty else []

    if raw.empty:
        st.warning("No backtest data found. Run: `python backtest_real_odds.py --all --edge 12`")
        st.stop()



    # Load saved settings — session state first, then config file, then hardcoded defaults
    try:
        from config.settings import MIN_EDGE_PCT, BET_MAX_ODDS, BET_MIN_ODDS, BET_MAX_EDGE
        cfg_min_edge = int(MIN_EDGE_PCT)
        cfg_max_edge = int(BET_MAX_EDGE)
        cfg_min_odds = int(abs(BET_MIN_ODDS))
        cfg_max_odds = int(BET_MAX_ODDS)
    except:
        cfg_min_edge, cfg_max_edge = 15, 30
        cfg_min_odds, cfg_max_odds = 140, 500

    saved_min_edge = st.session_state.get("saved_min_edge", cfg_min_edge)
    saved_max_edge = st.session_state.get("saved_max_edge", cfg_max_edge)
    saved_min_odds = st.session_state.get("saved_min_odds", cfg_min_odds)
    saved_max_odds = st.session_state.get("saved_max_odds", cfg_max_odds)

    # ── Filter controls ───────────────────────────────────────────────────────
    st.subheader("📐 Filter Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Edge**")
        min_edge = st.slider("Min edge %", 4, 30, saved_min_edge, step=1)
        max_edge = st.slider("Max edge %", 15, 60, saved_max_edge, step=1)

    with col2:
        st.markdown("**Odds range**")
        min_odds_abs = st.slider("Min odds (absolute, e.g. 140 = skip ≤+140)",
                                  100, 200, saved_min_odds, step=5)
        max_odds = st.slider("Max odds (e.g. 500 = skip >+500)",
                              200, 1000, saved_max_odds, step=25)
        underdogs_only = st.checkbox("Underdogs only", value=True)

    with col3:
        st.markdown("**Seasons**")
        if st.button("Select all seasons", key="sel_all"):
            st.session_state["sel_seasons"] = available_seasons
        selected_seasons = st.multiselect(
            "Include seasons",
            options=available_seasons,
            default=st.session_state.get("sel_seasons", available_seasons),
            key="sel_seasons_widget",
        )
        if not selected_seasons:
            selected_seasons = available_seasons
        show_clean_only = st.checkbox("Show clean seasons only in summary", value=True)

    min_odds_filter = -min_odds_abs  # convert to signed

    # ── Run filter ────────────────────────────────────────────────────────────
    filtered = apply_filters(
        raw, min_edge, max_edge, min_odds_filter, max_odds,
        underdogs_only, selected_seasons
    )

    # Current config for comparison
    current = apply_filters(raw, 15, 30, -140, 500, True, selected_seasons)

    st.divider()

    # ── Results summary ───────────────────────────────────────────────────────
    st.subheader("📊 Results")

    new_s  = summarise(filtered)
    curr_s = summarise(current)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total bets",    new_s["bets"],
              delta=f"{new_s['bets']-curr_s['bets']:+d} vs current")
    k2.metric("Bets / season", f"{new_s['bets']/max(len(selected_seasons),1):.0f}",
              delta=f"{(new_s['bets']-curr_s['bets'])/max(len(selected_seasons),1):+.0f}")
    k3.metric("Win rate",      f"{new_s['wr']:.1f}%",
              delta=f"{new_s['wr']-curr_s['wr']:+.1f}%")
    k4.metric("ROI",           f"{new_s['roi']:+.1f}%",
              delta=f"{new_s['roi']-curr_s['roi']:+.1f}%")
    k5.metric("Total P&L",     f"{new_s['pnl']:+.1f}u",
              delta=f"{new_s['pnl']-curr_s['pnl']:+.1f}u")

    if new_s["bets"] == 0:
        st.warning("No bets match these filters — try loosening the settings.")


        st.stop()

    # ── Per-season breakdown ──────────────────────────────────────────────────
    st.subheader("Per-Season Breakdown")

    season_rows = []
    for s in selected_seasons:
        s_df = filtered[filtered["season"] == s] if not filtered.empty else pd.DataFrame()
        if s_df.empty:
            season_rows.append({"Season": s, "Bets": 0, "Win%": "—",
                                 "ROI": "—", "P&L": "—", "Status": "no bets"})
            continue
        sm = summarise(s_df)
        status = "✅ clean" if s in CLEAN_SEASONS else "⚠️ training"
        season_rows.append({
            "Season": s, "Bets": sm["bets"],
            "Win%": f"{sm['wr']:.1f}%",
            "ROI":  f"{sm['roi']:+.1f}%",
            "P&L":  f"{sm['pnl']:+.1f}u",
            "Status": status,
        })

    season_df = pd.DataFrame(season_rows)
    st.dataframe(season_df, use_container_width=True, hide_index=True)

    # ── ROI bar chart ─────────────────────────────────────────────────────────
    plot_data = [r for r in season_rows if r["ROI"] != "—"]
    if plot_data:
        seasons_p = [r["Season"] for r in plot_data]
        rois_p    = [float(r["ROI"].replace("%","").replace("+","")) for r in plot_data]
        colors    = ["#3B6D11" if s in CLEAN_SEASONS else "#888780" for s in seasons_p]

        fig = go.Figure(go.Bar(
            x=seasons_p, y=rois_p,
            marker_color=[c if r >= 0 else "#A32D2D"
                          for c, r in zip(colors, rois_p)],
            text=[f"{r:+.1f}%" for r in rois_p],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="ROI by season (green/gray = profitable, red = losing, blue = clean out-of-sample)",
            height=320, yaxis_title="ROI %",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        # Highlight clean seasons
        for i, s in enumerate(seasons_p):
            if s in CLEAN_SEASONS:
                fig.add_vrect(x0=i-0.4, x1=i+0.4, fillcolor="#185FA5",
                              opacity=0.1, line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # ── Clean seasons focus ───────────────────────────────────────────────────
    clean_df = filtered[filtered["season"].isin(CLEAN_SEASONS)] if not filtered.empty else pd.DataFrame()
    if not clean_df.empty and show_clean_only:
        clean_s = summarise(clean_df)
        st.info(f"🎯 **Clean out-of-sample only (2024-25 + 2025-26):** "
                f"{clean_s['bets']} bets | WR {clean_s['wr']:.1f}% vs BE {clean_s['be']:.1f}% | "
                f"ROI **{clean_s['roi']:+.1f}%** | P&L {clean_s['pnl']:+.1f}u")

    # ── Odds range heatmap ────────────────────────────────────────────────────
    st.subheader("ROI by Odds Range")
    if not filtered.empty:
        bins  = [(-9999,-300),(-300,-200),(-200,-140),(-140,-110),(-110,100),
                 (100,150),(150,200),(200,300),(300,500),(500,9999)]
        labels= ["<-300","-300/-200","-200/-140","-140/-110","-110/EV",
                 "+100/+150","+150/+200","+200/+300","+300/+500",">+500"]
        heat  = []
        for (lo,hi), lbl in zip(bins, labels):
            sub = filtered[(filtered["bet_odds"]>=lo) & (filtered["bet_odds"]<hi)]
            if len(sub) < 3: continue
            sm  = summarise(sub)
            heat.append({"Range": lbl, "Bets": sm["bets"],
                         "Win%": sm["wr"], "ROI": sm["roi"]})

        if heat:
            hdf = pd.DataFrame(heat)
            fig2 = px.bar(hdf, x="Range", y="ROI", color="ROI",
                          color_continuous_scale=["#A32D2D","#BA7517","#3B6D11"],
                          text=hdf["ROI"].apply(lambda x: f"{x:+.1f}%"),
                          title="ROI by odds range (current filter settings)")
            fig2.update_layout(height=280, coloraxis_showscale=False,
                               plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)")
            fig2.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Save as current config button ─────────────────────────────────────────
    st.divider()
    if st.button("💾 Apply these filters as live config", type="primary"):
        # Save to session state (works on Streamlit Cloud where filesystem is read-only)
        st.session_state["saved_min_edge"]   = min_edge
        st.session_state["saved_max_edge"]   = max_edge
        st.session_state["saved_min_odds"]   = min_odds_abs
        st.session_state["saved_max_odds"]   = max_odds
        st.session_state["saved_underdogs"]  = underdogs_only

        # Also try writing to config file (works locally)
        try:
            import re
            cfg_path = ROOT / "config" / "settings.py"
            with open(cfg_path) as f:
                cfg = f.read()
            cfg = re.sub(r"MIN_EDGE_PCT\s*=\s*[\d.]+",
                         f"MIN_EDGE_PCT     = {float(min_edge)}", cfg)
            cfg = re.sub(r"BET_MAX_ODDS\s*=\s*[\d.]+",
                         f"BET_MAX_ODDS       = {int(max_odds)}", cfg)
            cfg = re.sub(r"BET_MIN_ODDS\s*=\s*-[\d.]+",
                         f"BET_MIN_ODDS       = {int(-min_odds_abs)}", cfg)
            cfg = re.sub(r"BET_MAX_EDGE\s*=\s*[\d.]+",
                         f"BET_MAX_EDGE       = {float(max_edge)}", cfg)
            with open(cfg_path, "w") as f:
                f.write(cfg)
        except:
            pass  # Read-only filesystem on cloud — session state is enough

        st.success(f"✅ Saved! Min edge: {min_edge}%, Odds: +{min_odds_abs+1}–+{max_odds}, "
                   f"Max edge: {max_edge}%. Sliders will default to these values this session.")
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: BET TRACKER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Bet Tracker":
    st.title("📋 Bet Tracker")

    bets = load_bets()

    with st.expander("➕ Log a new bet", expanded=len(bets) == 0):
        c1, c2, c3 = st.columns(3)
        with c1:
            b_date     = st.date_input("Date", value=date.today())
            b_home     = st.text_input("Home team", placeholder="LAC").upper()
            b_away     = st.text_input("Away team", placeholder="SAS").upper()
        with c2:
            b_bet_team = st.text_input("Bet on", placeholder="SAS").upper()
            b_odds     = st.number_input("Odds (American)", value=155, step=5)
            b_edge     = st.number_input("Edge %", value=20.0, step=0.5)
        with c3:
            b_units    = st.number_input("Units staked", value=1.0, step=0.5)
            b_result   = st.selectbox("Result", ["pending","win","loss"])

        if st.button("Log Bet", type="primary"):
            dec = american_to_decimal(b_odds)
            pnl = None
            if b_result == "win":   pnl = round((dec-1)*b_units, 3)
            if b_result == "loss":  pnl = -b_units
            bets.append({
                "date": b_date.isoformat(), "home_team": b_home,
                "away_team": b_away, "bet_team": b_bet_team,
                "bet_odds": fmt_ml(b_odds), "edge_pct": b_edge,
                "result": b_result, "units": b_units, "pnl": pnl,
            })
            save_bets(bets)
            st.success("Logged!")
            st.rerun()

    if not bets:
        st.info("No bets yet.")
    else:
        df = pd.DataFrame(bets)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False)
        decided = df[df["result"].isin(["win","loss"])]

        if len(decided):
            w = (decided["result"]=="win").sum()
            t = len(decided)
            pnl = decided["pnl"].sum()
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Total", len(df))
            m2.metric("Decided", t)
            m3.metric("Win Rate", f"{w/t:.1%}")
            m4.metric("P&L", f"{pnl:+.2f}u")
            m5.metric("ROI", f"{pnl/t*100:+.1f}%")
            st.divider()

        # Pending
        pending = df[df["result"]=="pending"]
        if len(pending):
            st.subheader(f"⏳ Pending ({len(pending)})")
            for i, (idx, row) in enumerate(pending.iterrows()):
                c1,c2,c3 = st.columns([4,2,2])
                with c1:
                    st.write(f"**{row['bet_team']} ({row['bet_odds']})** — "
                             f"{row['home_team']} vs {row['away_team']}  "
                             f"· {row['date'].strftime('%b %d')}")
                with c2:
                    new_result = st.selectbox("", ["pending","win","loss"],
                                              key=f"r_{i}",
                                              index=["pending","win","loss"].index(row["result"]))
                with c3:
                    if st.button("Update", key=f"u_{i}") and new_result != "pending":
                        all_bets = load_bets()
                        # Find matching bet
                        for b in all_bets:
                            if (b["date"] == row["date"].date().isoformat() and
                                b["bet_team"] == row["bet_team"] and
                                b["result"] == "pending"):
                                dec = american_to_decimal(b["bet_odds"])
                                b["result"] = new_result
                                b["pnl"] = round((dec-1)*float(b["units"]),3) if new_result=="win" else -float(b["units"])
                                break
                        save_bets(all_bets)
                        st.rerun()
            st.divider()

        # All bets table
        display = df[["date","home_team","away_team","bet_team","bet_odds","edge_pct","units","result","pnl"]].copy()
        display["date"] = display["date"].dt.strftime("%b %d")
        display.columns = ["Date","Home","Away","Bet","Odds","Edge%","Units","Result","P&L"]
        st.dataframe(display, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Performance":
    st.title("📈 Live Performance")

    bets = load_bets()
    if not bets:
        st.info("No bets tracked yet.")
        st.stop()

    df = pd.DataFrame(bets)
    df["date"] = pd.to_datetime(df["date"])
    df["pnl"]  = pd.to_numeric(df["pnl"], errors="coerce")
    decided = df[df["result"].isin(["win","loss"])].sort_values("date")

    if decided.empty:
        st.info("No decided bets yet.")
        st.stop()

    wins = (decided["result"]=="win").sum()
    total = len(decided)
    pnl   = decided["pnl"].sum()
    roi   = pnl/total*100

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Bets", total)
    k2.metric("Win Rate", f"{wins/total:.1%}")
    k3.metric("P&L", f"{pnl:+.2f}u")
    k4.metric("ROI", f"{roi:+.1f}%")
    k5.metric("Pending", (df["result"]=="pending").sum())

    decided["cum_pnl"] = decided["pnl"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decided["date"], y=decided["cum_pnl"],
        mode="lines+markers", line=dict(color="#3B6D11", width=2),
        marker=dict(size=8,
            color=decided["result"].map({"win":"#3B6D11","loss":"#A32D2D"})),
        hovertemplate="%{x|%b %d}<br>%{y:+.2f}u<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(title="Cumulative P&L", height=350,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Monthly
    decided["month"] = decided["date"].dt.to_period("M").astype(str)
    monthly = decided.groupby("month").agg(
        bets=("pnl","count"), pnl=("pnl","sum"),
        wins=("result", lambda x: (x=="win").sum())
    ).reset_index()
    monthly["roi"] = monthly["pnl"]/monthly["bets"]*100

    c1,c2 = st.columns(2)
    with c1:
        fig2 = px.bar(monthly, x="month", y="pnl",
                      color="pnl", color_continuous_scale=["#A32D2D","#BA7517","#3B6D11"],
                      title="Monthly P&L")
        fig2.update_layout(height=260, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.bar(monthly, x="month", y="roi",
                      color="roi", color_continuous_scale=["#A32D2D","#BA7517","#3B6D11"],
                      title="Monthly ROI %")
        fig3.update_layout(height=260, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: BACKTEST RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Backtest Results":
    st.title("📊 Historical Backtest Results")
    st.caption("Real closing lines from DraftKings/FanDuel/Caesars via SBR")

    raw = load_all_backtest()
    if raw.empty:
        st.warning("No backtest data. Run: `python backtest_real_odds.py --all --edge 15`")
        st.stop()

    # Use current settings to evaluate
    try:
        from config.settings import MIN_EDGE_PCT, BET_MAX_ODDS, BET_MIN_ODDS, BET_MAX_EDGE
    except:
        MIN_EDGE_PCT, BET_MAX_ODDS, BET_MIN_ODDS, BET_MAX_EDGE = 15.0, 500, -140, 30.0
    filtered = apply_filters(raw, MIN_EDGE_PCT, BET_MAX_EDGE,
                             BET_MIN_ODDS, BET_MAX_ODDS, True,
                             sorted(raw["season"].unique().tolist()))

    summary = []
    for s in sorted(raw["season"].unique()):
        s_df = filtered[filtered["season"]==s] if not filtered.empty else pd.DataFrame()
        if s_df.empty: continue
        sm = summarise(s_df)
        summary.append({
            "Season": s, "Bets": sm["bets"],
            "Win Rate": f"{sm['wr']:.1f}%",
            "Break-even": f"{sm['be']:.1f}%",
            "ROI": f"{sm['roi']:+.1f}%",
            "P&L": f"{sm['pnl']:+.1f}u",
            "Out-of-Sample": "✅ Clean" if s in CLEAN_SEASONS else "⚠️ Training",
        })

    if summary:
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        seasons_p = [r["Season"] for r in summary]
        rois_p    = [float(r["ROI"].replace("%","").replace("+","")) for r in summary]
        colors    = ["#185FA5" if s in CLEAN_SEASONS else
                     ("#3B6D11" if r >= 0 else "#A32D2D")
                     for s,r in zip(seasons_p, rois_p)]

        fig = go.Figure(go.Bar(
            x=seasons_p, y=rois_p,
            marker_color=colors,
            text=[f"{r:+.1f}%" for r in rois_p],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="ROI by Season  (blue = clean out-of-sample  |  green/red = training era)",
            height=360, yaxis_title="ROI %",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("✅ 2024-25 and 2025-26 are fully out-of-sample — these are the honest numbers. "
                "Earlier seasons were in the model's training data and results are inflated.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: MODEL STATS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Model Stats":
    st.title("⚙️ Model Stats & Calibration")

    raw = load_all_backtest()
    clean = raw[raw["season"].isin(CLEAN_SEASONS)].copy() if not raw.empty else pd.DataFrame()

    # Strategy overview
    try:
        from config.settings import MIN_EDGE_PCT, BET_MAX_ODDS, BET_MIN_ODDS, BET_MAX_EDGE
    except:
        MIN_EDGE_PCT, BET_MAX_ODDS, BET_MIN_ODDS, BET_MAX_EDGE = 15.0, 500, -140, 30.0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Min edge",     f"{MIN_EDGE_PCT}%")
    c2.metric("Odds range",   f"+{abs(BET_MIN_ODDS)+1} to +{BET_MAX_ODDS}")
    c3.metric("Max edge cap", f"{BET_MAX_EDGE}%")
    c4.metric("Bet type",     "Underdogs only")

    st.divider()

    if clean.empty:
        st.info("No clean season data found.")
        st.stop()

    clean["model_prob_home"] = pd.to_numeric(clean["model_prob_home"], errors="coerce")
    clean["home_won"]        = pd.to_numeric(clean.get("home_won", clean.get("home_actually_won")), errors="coerce")
    clean = clean.dropna(subset=["model_prob_home","home_won"])

    # Calibration
    st.subheader("Calibration (clean seasons only)")
    bins = np.linspace(0.25, 0.85, 8)
    cal  = []
    for i in range(len(bins)-1):
        mask = (clean["model_prob_home"]>=bins[i]) & (clean["model_prob_home"]<bins[i+1])
        sub  = clean[mask]
        if len(sub) < 10: continue
        cal.append({
            "bin":    f"{bins[i]:.0%}–{bins[i+1]:.0%}",
            "model":  sub["model_prob_home"].mean(),
            "actual": sub["home_won"].mean(),
            "n":      len(sub),
        })

    if cal:
        cal_df = pd.DataFrame(cal)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cal_df["model"], y=cal_df["actual"],
            mode="markers+lines",
            marker=dict(size=cal_df["n"].apply(lambda x: max(8,min(24,x//5))),
                        color="#185FA5"),
            hovertemplate="Model: %{x:.1%}<br>Actual: %{y:.1%}<extra></extra>",
            name="Calibration",
        ))
        fig.add_trace(go.Scatter(
            x=[0.25,0.85], y=[0.25,0.85],
            mode="lines", line=dict(dash="dot", color="gray"),
            name="Perfect calibration",
        ))
        fig.update_layout(
            xaxis=dict(tickformat=".0%", title="Model probability"),
            yaxis=dict(tickformat=".0%", title="Actual win rate"),
            height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Points below the diagonal = model overconfident. Bubble size = number of games.")

    # Architecture
    st.subheader("Model Architecture")
    arch = {
        "Ensemble":        "LR (25%) + XGBoost (35%) + LightGBM (35%) + Elo (5%)",
        "Features":        "138 (rolling stats, Elo, eFG%, rest/schedule)",
        "Training data":   "5 seasons: 2018-19 to 2022-23",
        "Validation":      "2023-24 (early stopping)",
        "Test set":        "2024-25 + 2025-26 (held out)",
        "Calibration":     "Isotonic regression on validation set",
    }
    for k, v in arch.items():
        c1,c2 = st.columns([2,4])
        c1.write(f"**{k}**")
        c2.write(v)