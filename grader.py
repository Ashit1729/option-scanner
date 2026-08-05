"""
Weekly grader — runs Friday after the close.
Reads journal.csv and answers one question: is this system earning the
right to real money?

For every fired trigger it re-fetches that day's 5-min bars and checks
which came first AFTER the fire: the favorable move needed for roughly a
+10% option gain, or a close through the invalidation level.

HONESTY NOTE: the P&L figures are a ROUGH PROXY. They assume delta ~0.40,
ignore spreads, slippage and theta, and use closes only. They are for
grading the SYSTEM's hit rate, not for bragging. Real results would be
worse than this proxy. If even the proxy can't stay positive over the
paper month, real money would have done worse.
"""

import datetime as dt
import os
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")
JOURNAL = "journal.csv"
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
LOOKBACK_DAYS = 7
ASSUMED_DELTA = 0.40
TARGET_PCT = 0.40          # take profit at +40% of premium
STOP_PCT = 0.20            # give up at -20% of premium

PAPER_START = dt.date(2026, 8, 3)
PAPER_WEEKS = 4


def discord(msg: str) -> None:
    if not WEBHOOK:
        print(msg)
        return
    for i in range(0, len(msg), 1900):
        requests.post(WEBHOOK, json={"content": msg[i:i + 1900]}, timeout=15)


def paper_status() -> str:
    week = ((dt.datetime.now(ET).date() - PAPER_START).days // 7) + 1
    if week < 1:
        return "Paper mode starts week of Aug 3."
    if week <= PAPER_WEEKS:
        return f"Paper week {week}/{PAPER_WEEKS} — still zero real orders."
    return ("Paper month complete. Go live ONLY if the last 4 weeks show a "
            "positive proxy P&L and a win rate you'd accept repeating.")


def grade_fired_row(r) -> dict:
    """WIN if the required favorable close happens before an invalidation
    close, LOSS if invalidation first, OPEN if neither by the close."""
    day = str(r["run_date"])[:10]
    try:
        start = dt.date.fromisoformat(day)
    except ValueError:
        return {"verdict": "ERR", "pnl": 0.0}
    df = yf.Ticker(r["ticker"]).history(
        start=start.isoformat(),
        end=(start + dt.timedelta(days=1)).isoformat(),
        interval="5m")
    if df is None or df.empty:
        return {"verdict": "ERR", "pnl": 0.0}
    idx = df.index
    idx = (idx.tz_convert("America/New_York") if idx.tz is not None
           else idx.tz_localize("America/New_York"))
    df = df.set_axis(idx)
    fired_time = str(r["fired_time"]) or "09:30"
    df = df[df.index.strftime("%H:%M") >= fired_time]
    if df.empty:
        return {"verdict": "ERR", "pnl": 0.0}

    ask = float(r["ask"])
    entry = float(r["fired_price"]) if str(r["fired_price"]) else float(r["trigger"])
    invalid = float(r["invalid"])
    is_call = r["direction"] == "CALL"
    req_move = TARGET_PCT * ask / ASSUMED_DELTA   # underlying move for ~+10%

    stop_move = STOP_PCT * ask / ASSUMED_DELTA   # underlying move for -20%
    for _, bar in df.iterrows():
        close = float(bar["Close"])
        fav = (close - entry) if is_call else (entry - close)
        chart_hit = (close < invalid) if is_call else (close > invalid)
        prem_hit = fav <= -stop_move
        if fav >= req_move:
            return {"verdict": "WIN", "pnl": TARGET_PCT * ask * 100}
        if chart_hit or prem_hit:
            adverse = min(-fav, stop_move) if prem_hit else (
                (entry - invalid) if is_call else (invalid - entry))
            loss = min(abs(adverse) * ASSUMED_DELTA * 100, ask * 100)
            return {"verdict": "LOSS", "pnl": -loss}
    # neither hit by close: mark to the day's last close
    last = float(df["Close"].iloc[-1])
    fav = (last - entry) if is_call else (entry - last)
    return {"verdict": "OPEN", "pnl": fav * ASSUMED_DELTA * 100}


def main() -> None:
    if not os.path.exists(JOURNAL):
        discord("📊 Weekly grade: no journal.csv yet — nothing to score.")
        return
    df = pd.read_csv(JOURNAL).fillna("")
    if df.empty:
        discord("📊 Weekly grade: journal is empty — nothing to score.")
        return

    cutoff = (dt.datetime.now(ET) - dt.timedelta(days=LOOKBACK_DAYS))
    df["day"] = df["run_date"].astype(str).str[:10]
    df = df[pd.to_datetime(df["day"], errors="coerce")
            >= pd.Timestamp(cutoff.date())]

    scans = df["day"].nunique()
    sitouts = int((df["setup"] == "no_setup").sum())
    cands = df[(df["ticker"] != "-") & (df["ticker"] != "")]
    fits = int((cands["fits_cap"].astype(str) == "True").sum())
    fired = cands[cands["trigger_fired"].astype(str) == "Y"]

    lines = [f"📊 **Weekly grade — last {LOOKBACK_DAYS} days**",
             f"Scan days: {scans} | sit-out days: {sitouts} | "
             f"candidates: {len(cands)} | fit the $ cap: {fits} | "
             f"triggers fired: {len(fired)}"]

    wins = losses = opens = 0
    pnl = 0.0
    for _, r in fired.iterrows():
        g = grade_fired_row(r)
        if g["verdict"] == "WIN":
            wins += 1
        elif g["verdict"] == "LOSS":
            losses += 1
        elif g["verdict"] == "OPEN":
            opens += 1
        pnl += g["pnl"]
        lines.append(f"• {r['day']} {r['ticker']} {r['direction']} "
                     f"@ {r['fired_price'] or r['trigger']} → {g['verdict']} "
                     f"({g['pnl']:+.0f}$ proxy)")

    if len(fired):
        lines.append(f"**Record: {wins}W / {losses}L / {opens} open. "
                     f"Proxy P&L: {pnl:+.0f}$ per 1-contract basis.**")
        lines.append("Proxy assumes delta 0.40, closes only, no spreads or "
                     "theta — real results would be worse.")
    else:
        lines.append("No triggers fired this week. Patience logged. "
                     "A quiet week costs nothing; a forced trade does.")

    lines.append(f"🧾 {paper_status()}")
    discord("\n".join(lines))


if __name__ == "__main__":
    main()
