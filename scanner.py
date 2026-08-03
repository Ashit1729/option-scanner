"""
Daily options setup scanner + intraday trigger watcher — v2
Phase 1 (scan, ~9:50 ET): find setups, post trade cards to Discord.
Phase 2 (watch, until 10:35 ET): poll 5-min candles; ping the moment a
  trigger CLOSES through its level on above-average volume, ping again if
  the invalidation closes, and post the 10:30 time-stop reminder.
Phase 3: journal every candidate WITH its outcome for the weekly grader.

It never places trades. Free Yahoo data can lag ~15 min on option quotes;
5-min price bars are near-real-time but not guaranteed. Your broker price
alert (set from the morning card) stays the primary real-time layer.
"""

import csv
import datetime as dt
import os
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")

# ----------------------------- CONFIG ---------------------------------------
WATCHLIST = ["SPY", "QQQ", "IWM", "AAPL", "AMZN", "NVDA",
             "MSFT", "TSLA", "META", "GOOGL"]

MAX_RISK = 100.0          # worst case on a long option = full premium
MAX_CANDIDATES = 2

# Zone detection (Lesson 3)
PIVOT_WINGS = 2
ZONE_CLUSTER_PCT = 0.006
MIN_TOUCHES = 2
NEAR_PCT = 0.012

# Liquidity gates (Lesson 2)
MIN_OI = 500
MIN_VOL_WARN = 100
MAX_SPREAD_PCT = 0.05

# Contract selection (Lesson 1)
MIN_DTE, MAX_DTE = 5, 12

# Paper period
PAPER_START = dt.date(2026, 8, 3)
PAPER_WEEKS = 4

# Macro stand-down days (verified 2026-07-31). UPDATE MONTHLY:
#   CPI  -> bls.gov/schedule/news_release/cpi.htm
#   FOMC -> federalreserve.gov/monetarypolicy/fomccalendars.htm
MACRO_EVENTS = {
    "2026-08-12": "CPI release 8:30 ET",
    "2026-09-16": "FOMC decision 2:00 ET",
    "2026-10-28": "FOMC decision 2:00 ET",
    "2026-12-09": "FOMC decision 2:00 ET",
}

# Timing (all ET)
GATE_START = (9, 35)      # scan may start from here...
GATE_END = (10, 20)       # ...until here (covers late GitHub starts)
TIME_STOP = (10, 30)      # your decision deadline
WATCH_END = (10, 35)      # watcher hard stop
CHECK_EVERY = 150         # seconds between polls

JOURNAL = "journal.csv"
JOURNAL_FIELDS = ["run_date", "ticker", "direction", "setup", "price",
                  "zone_lo", "zone_hi", "trigger", "invalid", "expiry",
                  "strike", "ask", "spread_pct", "oi", "vol", "iv_pct",
                  "cost", "fits_cap", "trigger_fired", "fired_time",
                  "fired_price", "invalidated", "note"]

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE_RUN = os.environ.get("FORCE_RUN", "0") == "1"
WATCH = os.environ.get("WATCH", "0") == "1"
# -----------------------------------------------------------------------------


def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def mins_now() -> int:
    t = now_et()
    return t.hour * 60 + t.minute


def within_gate() -> bool:
    return (GATE_START[0] * 60 + GATE_START[1]
            <= mins_now()
            <= GATE_END[0] * 60 + GATE_END[1])


def discord(msg: str) -> None:
    if not WEBHOOK:
        print("No DISCORD_WEBHOOK_URL set; printing instead:\n", msg)
        return
    for i in range(0, len(msg), 1900):
        try:
            requests.post(WEBHOOK, json={"content": msg[i:i + 1900]}, timeout=15)
        except Exception as e:
            print("Discord post failed:", e)
        time.sleep(0.5)


def find_zones(df: pd.DataFrame) -> list:
    highs, lows = df["High"].values, df["Low"].values
    pivots = []
    w = PIVOT_WINGS
    for i in range(w, len(df) - w):
        if highs[i] == max(highs[i - w:i + w + 1]):
            pivots.append(float(highs[i]))
        if lows[i] == min(lows[i - w:i + w + 1]):
            pivots.append(float(lows[i]))
    pivots.sort()
    zones = []
    for p in pivots:
        if zones and p <= zones[-1]["hi"] * (1 + ZONE_CLUSTER_PCT):
            zones[-1]["hi"] = max(zones[-1]["hi"], p)
            zones[-1]["touches"] += 1
        else:
            zones.append({"lo": p, "hi": p, "touches": 1})
    return [z for z in zones if z["touches"] >= MIN_TOUCHES]


def scan_ticker(sym: str):
    t = yf.Ticker(sym)
    df = t.history(period="6mo", interval="1d")
    if df is None or len(df) < 60:
        return None, "no data"

    price = float(df["Close"].iloc[-1])
    ema20 = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(df["Close"].ewm(span=50, adjust=False).mean().iloc[-1])

    if price > ema20 > ema50:
        trend = "up"
    elif price < ema20 < ema50:
        trend = "down"
    else:
        return None, "mixed trend (chop filter)"

    best = None
    for z in find_zones(df):
        if trend == "up" and z["hi"] >= price:
            dist = (z["hi"] - price) / price
            if dist <= NEAR_PCT:
                cand = {"direction": "CALL", "setup": "breakout watch",
                        "trigger": z["hi"], "invalid": z["lo"], **z, "dist": dist}
                if best is None or (cand["touches"], -cand["dist"]) > (best["touches"], -best["dist"]):
                    best = cand
        if trend == "down" and z["lo"] <= price:
            dist = (price - z["lo"]) / price
            if dist <= NEAR_PCT:
                cand = {"direction": "PUT", "setup": "breakdown watch",
                        "trigger": z["lo"], "invalid": z["hi"], **z, "dist": dist}
                if best is None or (cand["touches"], -cand["dist"]) > (best["touches"], -best["dist"]):
                    best = cand
    if best is None:
        return None, f"{trend}trend, no tested zone within {NEAR_PCT:.1%}"

    best.update({"ticker": sym, "price": price, "trend": trend,
                 "fired": False, "invalidated": False,
                 "fired_time": "", "fired_price": ""})
    return best, None


def earnings_within(t: yf.Ticker, days: int = 3):
    try:
        cal = t.calendar
        dates = []
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", []) or []
        elif cal is not None and hasattr(cal, "index"):
            dates = list(cal.loc["Earnings Date"]) if "Earnings Date" in cal.index else []
        today = now_et().date()
        for d in dates:
            d = d.date() if hasattr(d, "date") else d
            if 0 <= (d - today).days <= days:
                return True
        return False if dates else None
    except Exception:
        return None


def pick_contract(sym: str, direction: str, trigger: float):
    t = yf.Ticker(sym)
    today = now_et().date()
    expiry = dte = None
    for exp in (t.options or []):
        d = (dt.date.fromisoformat(exp) - today).days
        if MIN_DTE <= d <= MAX_DTE:
            expiry, dte = exp, d
            break
    if expiry is None:
        return None, "no expiry in 5-12 DTE window"

    chain = t.option_chain(expiry)
    df = (chain.calls if direction == "CALL" else chain.puts).copy()
    df["volume"] = df["volume"].fillna(0)
    df["openInterest"] = df["openInterest"].fillna(0)
    if direction == "CALL":
        df = df[df["strike"] >= trigger].sort_values("strike")
    else:
        df = df[df["strike"] <= trigger].sort_values("strike", ascending=False)

    for _, r in df.head(4).iterrows():
        bid, ask = float(r["bid"]), float(r["ask"])
        if bid <= 0 or ask <= 0:
            continue
        spread = ask - bid
        if spread > max(0.05, MAX_SPREAD_PCT * ask):
            continue
        if int(r["openInterest"]) < MIN_OI:
            continue
        return {"expiry": expiry, "dte": dte, "strike": float(r["strike"]),
                "bid": bid, "ask": ask, "spread_pct": spread / ask * 100,
                "oi": int(r["openInterest"]), "vol": int(r["volume"]),
                "iv_pct": float(r["impliedVolatility"] or 0) * 100,
                "cost": ask * 100}, None
    return None, "no strike passed liquidity gates"


def paper_status() -> str:
    week = ((now_et().date() - PAPER_START).days // 7) + 1
    if week < 1:
        return "PAPER MODE (starts week of Aug 3)"
    if week <= PAPER_WEEKS:
        return f"PAPER MODE — week {week}/{PAPER_WEEKS}. No real orders yet."
    return "Live-eligible — 1 contract max, only if the journal hit-rate justifies it."


def journal_row(row: dict) -> None:
    full = {k: row.get(k, "") for k in JOURNAL_FIELDS}
    exists = os.path.exists(JOURNAL)
    with open(JOURNAL, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(full)


def build_card(c: dict, k: dict, earn_note: str) -> str:
    fits = k["cost"] <= MAX_RISK
    size_line = (f"✅ 1 contract = ${k['cost']:.0f} premium at risk"
                 if fits else
                 f"❌ ${k['cost']:.0f} premium exceeds the ${MAX_RISK:.0f} cap → "
                 f"paper-log only, or wait for the spreads lesson. Do NOT size up.")
    vol_warn = ("" if k["vol"] >= MIN_VOL_WARN
                else " ⚠️ low volume so far today — recheck before entry.")
    side = "C" if c["direction"] == "CALL" else "P"
    above = "above" if c["direction"] == "CALL" else "below"
    below = "below" if c["direction"] == "CALL" else "above"
    return (
        f"**{c['ticker']} — {c['setup']} ({c['direction']})**  "
        f"[{c['trend']}trend]{earn_note}\n"
        f"Price {c['price']:.2f} | zone {c['lo']:.2f}–{c['hi']:.2f} "
        f"({c['touches']} touches)\n"
        f"Contract: {k['expiry']} ({k['dte']}DTE) ${k['strike']:.1f}{side} | "
        f"bid {k['bid']:.2f} / ask {k['ask']:.2f} (spread {k['spread_pct']:.1f}%) | "
        f"OI {k['oi']:,} | vol {k['vol']:,}{vol_warn} | IV {k['iv_pct']:.0f}%\n"
        f"{size_line}\n"
        f"📍 Set a broker/TradingView price alert at **{c['trigger']:.2f}** now "
        f"(backup layer).\n"
        f"Enter ONLY if a 5-min candle CLOSES {above} {c['trigger']:.2f} "
        f"on above-average volume.\n"
        f"Wrong the moment a 5-min candle closes back {below} "
        f"{c['invalid']:.2f} → exit.\n"
        f"At broker, confirm: delta 0.35–0.50, theta/day ≤ 8% of premium."
    )


def completed_5m_bars(sym: str):
    """Today's COMPLETED 5-min bars only (drops the still-forming bar)."""
    try:
        df = yf.Ticker(sym).history(period="1d", interval="5m")
        if df is None or df.empty:
            return None
        idx = df.index
        idx = (idx.tz_convert("America/New_York") if idx.tz is not None
               else idx.tz_localize("America/New_York"))
        df = df.set_axis(idx)
        ts_now = pd.Timestamp.now(tz="America/New_York")
        df = df[df.index + pd.Timedelta(minutes=5) <= ts_now]
        return df if not df.empty else None
    except Exception:
        return None


def watch_loop(cands: list) -> None:
    end_mins = WATCH_END[0] * 60 + WATCH_END[1]
    stop_mins = TIME_STOP[0] * 60 + TIME_STOP[1]
    pinged_timestop = False

    while mins_now() < end_mins:
        for c in cands:
            df = completed_5m_bars(c["ticker"])
            if df is None:
                continue
            bar = df.iloc[-1]
            close, vol = float(bar["Close"]), float(bar["Volume"])
            avg_vol = float(df["Volume"].mean())
            is_call = c["direction"] == "CALL"

            if not c["fired"]:
                hit = close > c["trigger"] if is_call else close < c["trigger"]
                if hit and vol >= avg_vol:
                    c["fired"] = True
                    c["fired_time"] = df.index[-1].strftime("%H:%M")
                    c["fired_price"] = round(close, 2)
                    discord(f"🔔 **{c['ticker']} TRIGGER FIRED** "
                            f"{c['fired_time']} ET — 5-min close {close:.2f} "
                            f"{'above' if is_call else 'below'} {c['trigger']:.2f} "
                            f"on {vol/avg_vol:.1f}x avg volume.\n"
                            f"Confirm the candle on YOUR chart, check the live "
                            f"chain, then decide. Invalid if a 5-min close goes "
                            f"back {'below' if is_call else 'above'} "
                            f"{c['invalid']:.2f}. {paper_status()}")
            elif not c["invalidated"]:
                bad = close < c["invalid"] if is_call else close > c["invalid"]
                if bad:
                    c["invalidated"] = True
                    discord(f"❌ **{c['ticker']} INVALIDATED** — 5-min close "
                            f"{close:.2f} through {c['invalid']:.2f}. "
                            f"The setup is wrong. If you entered, exit now. "
                            f"No revenge trades.")

        if not pinged_timestop and mins_now() >= stop_mins:
            pinged_timestop = True
            if any(c["fired"] for c in cands):
                discord("⏰ **10:30 ET time stop.** If a trade isn't working, "
                        "close it. If it is working, manage it — trail your "
                        "invalidation, don't invent new targets.")
            else:
                discord("⏰ 10:30 ET — no trigger fired. The day is over for "
                        "this system. Standing down.")
                return
        time.sleep(CHECK_EVERY)


def main() -> None:
    if not FORCE_RUN and not within_gate():
        print("Outside scan window; exiting (expected for one of the two crons).")
        return

    today = now_et().date()
    tstamp = now_et().strftime("%Y-%m-%d %H:%M ET")

    label = MACRO_EVENTS.get(today.isoformat())
    if label:
        discord(f"⚠️ **{today} — macro day: {label}.**\n"
                f"Stand down. No scan, no trades today.")
        return

    spy = yf.Ticker("SPY").history(period="5d", interval="1d")
    if spy is None or spy.empty or spy.index[-1].date() != today:
        if not FORCE_RUN:
            print("Market appears closed today; exiting quietly.")
            return
        print("Note: no fresh SPY bar; continuing because FORCE_RUN=1.")

    candidates, reasons = [], []
    for sym in WATCHLIST:
        try:
            cand, why = scan_ticker(sym)
            if cand is None:
                reasons.append(f"{sym}: {why}")
            else:
                candidates.append(cand)
        except Exception as e:
            reasons.append(f"{sym}: data error ({type(e).__name__})")
        time.sleep(1)

    candidates.sort(key=lambda c: (-c["touches"], c["dist"]))
    candidates = candidates[:MAX_CANDIDATES]

    kept = []
    cards = [f"🎯 **Scan {tstamp}** (data delayed — verify at broker)"]
    for c in candidates:
        earn = earnings_within(yf.Ticker(c["ticker"]))
        if earn is True:
            cards.append(f"⏭️ **{c['ticker']}** skipped: earnings within 3 days "
                         f"(IV crush risk).")
            continue
        earn_note = ("" if earn is False
                     else " ⚠️ earnings date unverified — check before trading.")
        k, why = pick_contract(c["ticker"], c["direction"], c["trigger"])
        if k is None:
            cards.append(f"⏭️ **{c['ticker']} {c['direction']}** setup found but "
                         f"no tradeable contract: {why}.")
            continue
        c["contract"] = k
        kept.append(c)
        cards.append(build_card(c, k, earn_note))

    if not kept:
        cards.append("🪑 **No tradeable setup today. Sit out.** That is the "
                     "system working, not failing.\n"
                     + "\n".join(f"• {r}" for r in reasons[:10]))
        discord("\n\n".join(cards))
        journal_row({"run_date": tstamp, "ticker": "-", "setup": "no_setup",
                     "note": "; ".join(reasons)[:400]})
        return

    if WATCH:
        cards.append(f"👁️ Watcher live until {WATCH_END[0]}:{WATCH_END[1]:02d} ET — "
                     f"I'll ping when a trigger fires. Broker alert stays the backup.")
    cards.append(f"🧾 {paper_status()}\nNothing here is financial advice or a "
                 f"prediction. It is your own checklist, automated.")
    discord("\n\n".join(cards))

    if WATCH:
        watch_loop(kept)

    for c in kept:
        k = c["contract"]
        journal_row({"run_date": tstamp, "ticker": c["ticker"],
                     "direction": c["direction"], "setup": c["setup"],
                     "price": round(c["price"], 2), "zone_lo": round(c["lo"], 2),
                     "zone_hi": round(c["hi"], 2),
                     "trigger": round(c["trigger"], 2),
                     "invalid": round(c["invalid"], 2), "expiry": k["expiry"],
                     "strike": k["strike"], "ask": k["ask"],
                     "spread_pct": round(k["spread_pct"], 1), "oi": k["oi"],
                     "vol": k["vol"], "iv_pct": round(k["iv_pct"], 1),
                     "cost": round(k["cost"], 0),
                     "fits_cap": k["cost"] <= MAX_RISK,
                     "trigger_fired": "Y" if c["fired"] else
                                      ("N" if WATCH else "not_watched"),
                     "fired_time": c["fired_time"],
                     "fired_price": c["fired_price"],
                     "invalidated": "Y" if c["invalidated"] else "N"})


if __name__ == "__main__":
    main()
