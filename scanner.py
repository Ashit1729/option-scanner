"""
Full-day options setup scanner — v3

Runs in two sessions because GitHub caps a single job at 6 hours and the
US session is 6.5:
    AM session  9:35 -> 12:40 ET
    PM session 12:50 -> 16:05 ET
State is handed from AM to PM via state.json so a setup found at 10:15
is still being watched at 14:00.

Each session loops:
  - every 30 min: rescan all 20 tickers for NEW setups -> post a trade card
  - every 2 min : poll 5-min bars on every active setup
                  -> ping when a trigger CLOSES through its level
                  -> ping when an entered setup CLOSES through invalidation

It never places trades. Yahoo option quotes can lag ~15 min; 5-min bars
are near-real-time but not guaranteed. Your broker price alert, set from
the card, remains the primary real-time layer.
"""

import csv
import datetime as dt
import json
import math
import os
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")

# ----------------------------- CONFIG ---------------------------------------
# Top 20 S&P 500 companies by index weight, verified 2026-08-03 12:15 ET
# (chartrow.com, SEC N-PORT filing rolled forward on price).
# Recheck quarterly — this ordering drifts.
WATCHLIST = ["NVDA", "AAPL", "GOOGL", "MSFT", "AMZN",
             "AVGO", "META", "JPM", "BRK-B", "MU",
             "LLY", "TSLA", "AMD", "XOM", "JNJ",
             "V", "WMT", "MA", "CSCO", "ABBV", "SPY", "QQQ", "IWM"]

ACCOUNT_SIZE = 1000.0     # used only to show risk as a % of your account
MAX_RISK = 400.0          # worst case on a long option = the full premium
                          # NOTE: this now gates the ESTIMATED ENTRY cost,
                          # not the current ask, because you buy later.
DELTA_TARGET = 0.45       # preferred delta at entry
DELTA_MIN, DELTA_MAX = 0.30, 0.60
RISK_FREE = 0.04          # only used for the Black-Scholes estimate
MAX_NEW_PER_SCAN = 5      # new cards per 30-min rescan
MAX_ACTIVE = 15           # ceiling on simultaneously watched setups

# Zone detection (Lesson 3)
PIVOT_WINGS = 2
ZONE_CLUSTER_PCT = 0.006
MIN_TOUCHES = 2
NEAR_PCT = 0.02

# Liquidity gates (Lesson 2)
MIN_OI = 500
MIN_VOL_WARN = 100
MAX_SPREAD_PCT = 0.05

# Contract selection (Lesson 1)
MIN_DTE, MAX_DTE = 5, 30

# Paper period
PAPER_START = dt.date(2026, 8, 3)
PAPER_WEEKS = 4

# Macro stand-down days. UPDATE MONTHLY:
#   CPI/NFP -> bls.gov/schedule/news_release/
#   FOMC    -> federalreserve.gov/monetarypolicy/fomccalendars.htm
MACRO_EVENTS = {
    "2026-08-07": "Jobs report (NFP) 8:30 ET",
    "2026-08-12": "CPI release 8:30 ET",
    "2026-09-16": "FOMC decision 2:00 ET",
    "2026-10-28": "FOMC decision 2:00 ET",
    "2026-12-09": "FOMC decision 2:00 ET",
}

# Sessions (ET). Gap between them so AM can commit state before PM checks out.
SESSIONS = {"am": ((9, 30), (12, 40)),
            "pm": ((12, 45), (16, 5))}
START_GRACE = 180         # a session may only BEGIN within this many minutes
                          # of its window opening. Generous, because a run
                          # queued behind another job can start very late and
                          # should still work. Duplicate daylight-saving crons
                          # are filtered by wrong_dst_twin() instead.
LATE_START_WARN = 30      # tell Discord if a session starts this late

NO_NEW_AFTER = (15, 30)       # stop opening new ideas this late
OVERNIGHT_WARN_AFTER = (15, 0)
RESCAN_EVERY = 1800           # seconds between full watchlist rescans
CHECK_EVERY = 120             # seconds between 5-min bar polls

JOURNAL = "journal.csv"
STATE = "state.json"
JOURNAL_FIELDS = ["run_date", "session", "ticker", "direction", "setup",
                  "price", "zone_lo", "zone_hi", "trigger", "invalid",
                  "expiry", "strike", "ask", "spread_pct", "oi", "vol",
                  "iv_pct", "cost", "fits_cap", "trigger_fired", "fired_time",
                  "fired_price", "invalidated", "note"]

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
SESSION = os.environ.get("SESSION", "am").lower()
FORCE_RUN = os.environ.get("FORCE_RUN", "0") == "1"
# -----------------------------------------------------------------------------


def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def mins_now() -> int:
    t = now_et()
    return t.hour * 60 + t.minute


def hm(t) -> int:
    return t[0] * 60 + t[1]


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


def to_et(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    idx = (idx.tz_convert("America/New_York") if idx.tz is not None
           else idx.tz_localize("America/New_York"))
    return df.set_axis(idx)


def batch_history(symbols, period, interval):
    """One HTTP call for the whole watchlist. Returns {sym: DataFrame}."""
    try:
        raw = yf.download(symbols, period=period, interval=interval,
                          group_by="ticker", auto_adjust=False,
                          progress=False, threads=False)
    except Exception as e:
        print("batch download failed:", e)
        return {}
    out = {}
    for s in symbols:
        try:
            df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(how="all")
            if not df.empty:
                out[s] = df
        except Exception:
            continue
    return out


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs(S, K, T, sig, call=True, r=RISK_FREE):
    """Black-Scholes price, delta and per-CALENDAR-DAY theta.
    Yahoo gives no greeks, so we derive them from the chain's own IV.
    These are model estimates — your broker's numbers are authoritative."""
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0:
        intrinsic = max(0.0, (S - K) if call else (K - S))
        return {"price": intrinsic, "delta": 1.0 if intrinsic > 0 else 0.0,
                "theta": 0.0}
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    disc = math.exp(-r * T)
    if call:
        price = S * _ncdf(d1) - K * disc * _ncdf(d2)
        delta = _ncdf(d1)
        theta = (-S * _npdf(d1) * sig / (2 * math.sqrt(T))
                 - r * K * disc * _ncdf(d2)) / 365.0
    else:
        price = K * disc * _ncdf(-d2) - S * _ncdf(-d1)
        delta = _ncdf(d1) - 1.0
        theta = (-S * _npdf(d1) * sig / (2 * math.sqrt(T))
                 + r * K * disc * _ncdf(-d2)) / 365.0
    return {"price": price, "delta": delta, "theta": theta}


def project_entry_ask(ask, spot, trigger, K, T, sig, call):
    """You are told to WAIT for the trigger, so you buy after the stock has
    already moved. Scale today's ask by the model's price ratio between
    here and the trigger. Estimate only — never a quote."""
    here = bs(spot, K, T, sig, call)["price"]
    there = bs(trigger, K, T, sig, call)["price"]
    if here <= 0.01:
        return ask
    ratio = min(max(there / here, 1.0), 3.0)
    return ask * ratio


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


def setup_from_daily(sym: str, df: pd.DataFrame, live_price: float):
    """Find the best zone near the CURRENT price. Returns (cand, reason)."""
    if df is None or len(df) < 60:
        return None, "no data"
    price = live_price if live_price else float(df["Close"].iloc[-1])
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
                 "fired_time": "", "fired_price": "", "logged": False})
    return best, None


def earnings_within(sym: str, days: int = 3):
    """True / False / None(unknown). yfinance earnings data is unreliable."""
    try:
        cal = yf.Ticker(sym).calendar
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


def pick_contract(sym: str, direction: str, trigger: float, spot: float):
    """Score every liquid strike near the trigger and return the one whose
    delta AT ENTRY is closest to DELTA_TARGET, gated on the ESTIMATED entry
    cost rather than the current ask."""
    t = yf.Ticker(sym)
    today = now_et().date()
    expiries = []
    for exp in (t.options or []):
        try:
            d = (dt.date.fromisoformat(exp) - today).days
        except ValueError:
            continue
        if MIN_DTE <= d <= MAX_DTE:
            expiries.append((exp, d))
    if not expiries:
        return None, f"no expiry in {MIN_DTE}-{MAX_DTE} DTE window"

    call = direction == "CALL"
    cands, too_pricey = [], None
    for exp, dte in expiries[:4]:
        try:
            chain = t.option_chain(exp)
        except Exception:
            continue
        df = (chain.calls if call else chain.puts).copy()
        df["volume"] = df["volume"].fillna(0)
        df["openInterest"] = df["openInterest"].fillna(0)
        if call:
            df = df[df["strike"] >= trigger].sort_values("strike")
        else:
            df = df[df["strike"] <= trigger].sort_values("strike",
                                                         ascending=False)
        T = max(dte, 1) / 365.0
        for _, r in df.head(8).iterrows():
            bid, ask = float(r["bid"]), float(r["ask"])
            iv = float(r["impliedVolatility"] or 0)
            if bid <= 0 or ask <= 0 or iv <= 0:
                continue
            spread = ask - bid
            if spread > max(0.05, MAX_SPREAD_PCT * ask):
                continue
            if int(r["openInterest"]) < MIN_OI:
                continue
            K = float(r["strike"])
            g = bs(trigger, K, T, iv, call)        # greeks AT ENTRY
            entry_ask = project_entry_ask(ask, spot, trigger, K, T, iv, call)
            k = {"expiry": exp, "dte": dte, "strike": K,
                 "bid": bid, "ask": ask, "spread_pct": spread / ask * 100,
                 "oi": int(r["openInterest"]), "vol": int(r["volume"]),
                 "iv_pct": iv * 100, "cost": ask * 100,
                 "entry_ask": entry_ask, "entry_cost": entry_ask * 100,
                 "delta": g["delta"], "theta": g["theta"],
                 "theta_pct": (abs(g["theta"]) / max(entry_ask, 0.01)) * 100}
            if k["entry_cost"] > MAX_RISK:
                if too_pricey is None or k["entry_cost"] < too_pricey["entry_cost"]:
                    too_pricey = k
                continue
            cands.append(k)
        time.sleep(0.4)

    if not cands:
        if too_pricey:
            return None, (f"cheapest liquid contract would cost about "
                          f"${too_pricey['entry_cost']:.0f} at the trigger "
                          f"(> ${MAX_RISK:.0f} cap)")
        return None, "no strike passed liquidity gates"

    cands.sort(key=lambda c: abs(abs(c["delta"]) - DELTA_TARGET))
    best = cands[0]
    best["delta_ok"] = DELTA_MIN <= abs(best["delta"]) <= DELTA_MAX
    return best, None


def wrong_dst_twin() -> bool:
    """Each session has two crons, one for EDT and one for EST, because
    GitHub cron is always UTC. Exactly one is correct on any given day.
    The workflow passes the cron string through CRON_SCHEDULE; we compare
    its UTC hour against what the current Eastern offset implies."""
    cron = os.environ.get("CRON_SCHEDULE", "").strip()
    if not cron:
        return False                      # manual run: no cron to check
    try:
        cron_hour = int(cron.split()[1])
    except (IndexError, ValueError):
        return False
    is_dst = bool(now_et().dst())
    expected = {"am": 13, "pm": 16} if is_dst else {"am": 14, "pm": 17}
    want = expected.get(SESSION)
    if want is None or cron_hour == want:
        return False
    print(f"cron {cron!r} is the "
          f"{'EST' if is_dst else 'EDT'} twin but today is "
          f"{'EDT' if is_dst else 'EST'}; exiting so the correct one runs.")
    return True


def paper_status() -> str:
    week = ((now_et().date() - PAPER_START).days // 7) + 1
    if week < 1:
        return "PAPER MODE (starts week of Aug 3)"
    if week <= PAPER_WEEKS:
        return f"PAPER MODE — week {week}/{PAPER_WEEKS}. No real orders yet."
    return "Live-eligible — 1 contract max, only if the journal justifies it."


def migrate_journal() -> None:
    """v2 journals lack the 'session' column. Rewrite the header once so
    appended rows stay aligned instead of silently shifting."""
    if not os.path.exists(JOURNAL):
        return
    try:
        with open(JOURNAL, newline="") as f:
            rows = list(csv.reader(f))
        if not rows or rows[0] == JOURNAL_FIELDS:
            return
        old = rows[0]
        with open(JOURNAL, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            w.writeheader()
            for r in rows[1:]:
                d = dict(zip(old, r))
                w.writerow({k: d.get(k, "") for k in JOURNAL_FIELDS})
        print("journal.csv migrated to v3 schema")
    except Exception as e:
        print("journal migration skipped:", e)


def journal_row(row: dict) -> None:
    full = {k: row.get(k, "") for k in JOURNAL_FIELDS}
    exists = os.path.exists(JOURNAL)
    with open(JOURNAL, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(full)


def log_candidate(c: dict) -> None:
    if c.get("logged"):
        return
    k = c.get("contract", {})
    journal_row({"run_date": now_et().strftime("%Y-%m-%d %H:%M ET"),
                 "session": SESSION, "ticker": c["ticker"],
                 "direction": c["direction"], "setup": c["setup"],
                 "price": round(c["price"], 2), "zone_lo": round(c["lo"], 2),
                 "zone_hi": round(c["hi"], 2), "trigger": round(c["trigger"], 2),
                 "invalid": round(c["invalid"], 2), "expiry": k.get("expiry", ""),
                 "strike": k.get("strike", ""), "ask": k.get("ask", ""),
                 "spread_pct": round(k.get("spread_pct", 0), 1),
                 "oi": k.get("oi", ""), "vol": k.get("vol", ""),
                 "iv_pct": round(k.get("iv_pct", 0), 1),
                 "cost": round(k.get("cost", 0), 0),
                 "fits_cap": k.get("cost", 1e9) <= MAX_RISK,
                 "trigger_fired": "Y" if c["fired"] else "N",
                 "fired_time": c["fired_time"], "fired_price": c["fired_price"],
                 "invalidated": "Y" if c["invalidated"] else "N"})
    c["logged"] = True


def build_card(c: dict, k: dict, earn_note: str) -> str:
    """Readable card: a monospaced block for the numbers, plain text for
    the plan. Every price is labelled with WHEN it applies."""
    call = c["direction"] == "CALL"
    side = "Call" if call else "Put"
    arrow = "🟢" if call else "🔴"
    above, below = ("above", "below") if call else ("below", "above")

    exp_date = dt.date.fromisoformat(k["expiry"])
    exp_txt = exp_date.strftime("%a %d %b %Y")
    dist = (c["trigger"] - c["price"]) / c["price"] * 100
    inval = (c["invalid"] - c["price"]) / c["price"] * 100

    # stock move needed for +10% on the option, solved on the model
    T = max(k["dte"], 1) / 365.0
    tgt_prem = k["entry_ask"] * 1.10
    lo, hi = c["trigger"], c["trigger"] * (1.15 if call else 0.85)
    for _ in range(50):
        mid = (lo + hi) / 2
        px = bs(mid, k["strike"], T, k["iv_pct"] / 100, call)["price"]
        if (px < tgt_prem) == call:
            lo = mid
        else:
            hi = mid
    tgt_stock = hi
    tgt_move = (tgt_stock - c["trigger"]) / c["trigger"] * 100

    pct_acct = k["entry_cost"] / ACCOUNT_SIZE * 100
    delta_flag = "" if k.get("delta_ok", True) else "  ⚠️ outside 0.30-0.60"
    theta_flag = "" if k["theta_pct"] <= 8 else "  ⚠️ heavy decay"
    vol_flag = "" if k["vol"] >= MIN_VOL_WARN else "  ⚠️ thin today"

    block = (
        f"STOCK\n"
        f"  price now      ${c['price']:>9.2f}\n"
        f"  enter above    ${c['trigger']:>9.2f}   {dist:+.2f}% away\n"
        f"  wrong below    ${c['invalid']:>9.2f}   {inval:+.2f}%\n"
        f"  zone           {c['lo']:.2f} - {c['hi']:.2f}   {c['touches']} touches\n"
        f"\n"
        f"CONTRACT   ${k['strike']:.0f} {side}\n"
        f"  expires        {exp_txt}   ({k['dte']} days)\n"
        f"  ask right now  ${k['ask']:>9.2f}   = ${k['cost']:.0f}\n"
        f"  est. AT ENTRY  ${k['entry_ask']:>9.2f}   = ${k['entry_cost']:.0f}"
        f"   <- what you pay\n"
        f"  bid / spread   ${k['bid']:.2f} / {k['spread_pct']:.1f}%\n"
        f"  OI / volume    {k['oi']:,} / {k['vol']:,}{vol_flag}\n"
        f"  IV             {k['iv_pct']:.0f}%\n"
        f"  delta          {k['delta']:+.2f}{delta_flag}\n"
        f"  theta          -${abs(k['theta']) * 100:.0f}/day "
        f"({k['theta_pct']:.0f}% of premium){theta_flag}\n"
        f"\n"
        f"RISK\n"
        f"  est. cost      ${k['entry_cost']:.0f}   = {pct_acct:.0f}% of "
        f"${ACCOUNT_SIZE:,.0f}\n"
        f"  max loss       ${k['entry_cost']:.0f}   (full premium on a gap)\n"
        f"\n"
        f"+10% TARGET\n"
        f"  option         ${tgt_prem:.2f}   = ${tgt_prem * 100:.0f}\n"
        f"  needs stock    ${tgt_stock:.2f}   {tgt_move:+.2f}% past trigger\n"
    )

    late = ("\n🌙 Late session — holding past 16:00 risks an overnight gap, "
            "which can cost the FULL premium regardless of your stop."
            if mins_now() >= hm(OVERNIGHT_WARN_AFTER) else "")

    return (
        f"{arrow} **{c['ticker']} · {c['direction']} · {c['setup']}** "
        f"· {c['trend']}trend{earn_note}\n"
        f"```\n{block}```\n"
        f"📍 **Set a broker alert at {c['trigger']:.2f} now.**\n"
        f"✅ Enter only if a 5-min candle CLOSES {above} "
        f"{c['trigger']:.2f} on above-average volume.\n"
        f"🛑 Wrong the moment a 5-min candle closes {below} "
        f"{c['invalid']:.2f} — exit.\n"
        f"🔍 Greeks above are Black-Scholes estimates from the chain's IV, "
        f"and the entry price is a projection. Confirm both on your broker's "
        f"live chain before you buy.{late}"
    )


def rescan(seen: set, active_count: int) -> list:
    """Full watchlist sweep. Returns new candidate dicts with contracts."""
    daily = batch_history(WATCHLIST, "6mo", "1d")
    intraday = batch_history(WATCHLIST, "1d", "5m")
    found = []
    for sym in WATCHLIST:
        if active_count + len(found) >= MAX_ACTIVE or len(found) >= MAX_NEW_PER_SCAN:
            break
        df = daily.get(sym)
        if df is None:
            continue
        live = None
        idf = intraday.get(sym)
        if idf is not None and not idf.empty:
            live = float(idf["Close"].iloc[-1])
        try:
            c, _ = setup_from_daily(sym, df, live)
        except Exception:
            continue
        if c is None:
            continue
        key = (c["ticker"], c["direction"], round(c["trigger"], 2))
        if key in seen:
            continue

        earn = earnings_within(sym)
        if earn is True:
            seen.add(key)
            continue
        c["earn_note"] = ("" if earn is False
                          else " ⚠️ earnings date unverified — check before trading.")
        k, why = pick_contract(sym, c["direction"], c["trigger"], c["price"])
        if k is None:
            seen.add(key)      # don't retry a rejected setup every 30 min
            c["reject"] = why
            found.append(c)    # reported as a skip line, not watched
            continue
        c["contract"] = k
        found.append(c)
        seen.add(key)
    return found


def poll(active: list) -> None:
    """One batched 5-min bar fetch, then check every active setup."""
    syms = sorted({c["ticker"] for c in active if not c["invalidated"]})
    if not syms:
        return
    bars = batch_history(syms, "1d", "5m")
    ts_now = pd.Timestamp.now(tz="America/New_York")
    for c in active:
        if c["invalidated"] or "contract" not in c:
            continue
        df = bars.get(c["ticker"])
        if df is None or df.empty:
            continue
        try:
            df = to_et(df)
            df = df[df.index + pd.Timedelta(minutes=5) <= ts_now]
        except Exception:
            continue
        if df.empty:
            continue
        bar = df.iloc[-1]
        close, vol = float(bar["Close"]), float(bar["Volume"])
        avg_vol = float(df["Volume"].mean())
        is_call = c["direction"] == "CALL"

        if not c["fired"]:
            if mins_now() >= hm(NO_NEW_AFTER):
                continue
            hit = close > c["trigger"] if is_call else close < c["trigger"]
            if hit and vol >= avg_vol:
                c["fired"] = True
                c["fired_time"] = df.index[-1].strftime("%H:%M")
                c["fired_price"] = round(close, 2)
                discord(f"🔔 **{c['ticker']} TRIGGER FIRED** {c['fired_time']} ET "
                        f"— 5-min close {close:.2f} "
                        f"{'above' if is_call else 'below'} {c['trigger']:.2f} "
                        f"on {vol / max(avg_vol, 1):.1f}x avg volume.\n"
                        f"Contract: {c['contract']['expiry']} "
                        f"${c['contract']['strike']:.1f}"
                        f"{'C' if is_call else 'P'} "
                        f"(~${c['contract']['cost']:.0f}).\n"
                        f"Confirm the candle on YOUR chart and check the live "
                        f"chain before anything else. Invalid if a 5-min close "
                        f"goes back {'below' if is_call else 'above'} "
                        f"{c['invalid']:.2f}. {paper_status()}")
        else:
            bad = close < c["invalid"] if is_call else close > c["invalid"]
            if bad:
                c["invalidated"] = True
                discord(f"❌ **{c['ticker']} INVALIDATED** — 5-min close "
                        f"{close:.2f} through {c['invalid']:.2f}. "
                        f"The setup is wrong. If you entered, exit. "
                        f"No revenge trades, no averaging down.")
                log_candidate(c)


def save_state(active: list) -> None:
    keep = [{k: v for k, v in c.items() if k != "earn_note"}
            for c in active if not c["invalidated"] and "contract" in c]
    try:
        with open(STATE, "w") as f:
            json.dump({"date": now_et().date().isoformat(),
                       "active": keep}, f)
    except Exception as e:
        print("state save failed:", e)


def load_state() -> list:
    try:
        with open(STATE) as f:
            data = json.load(f)
        if data.get("date") != now_et().date().isoformat():
            return []
        return data.get("active", [])
    except Exception:
        return []


def main() -> None:
    if SESSION not in SESSIONS:
        print("Unknown SESSION:", SESSION)
        return
    start, end = SESSIONS[SESSION]
    if not FORCE_RUN and wrong_dst_twin():
        return

    latest_start = min(hm(start) + START_GRACE, hm(end))
    if not FORCE_RUN and not (hm(start) <= mins_now() <= latest_start):
        print(f"{SESSION}: now {now_et():%H:%M} ET, may only start between "
              f"{hm(start)//60:02d}:{hm(start)%60:02d} and "
              f"{latest_start//60:02d}:{latest_start%60:02d}. Exiting "
              f"(expected for one of the two daylight-saving crons).")
        return

    migrate_journal()

    today = now_et().date()
    label = MACRO_EVENTS.get(today.isoformat())
    if label:
        if SESSION == "am":
            discord(f"⚠️ **{today} — macro day: {label}.**\n"
                    f"Stand down. No scan, no trades today.")
        return

    spy = yf.Ticker("SPY").history(period="5d", interval="1d")
    if spy is None or spy.empty or spy.index[-1].date() != today:
        if not FORCE_RUN:
            print("Market appears closed; exiting quietly.")
            return
        print("No fresh SPY bar; continuing because FORCE_RUN=1.")

    active = load_state() if SESSION == "pm" else []
    seen = {(c["ticker"], c["direction"], round(c["trigger"], 2)) for c in active}

    late = mins_now() - hm(start)
    if late >= LATE_START_WARN and not FORCE_RUN:
        discord(f"⏳ Heads up: the {SESSION.upper()} session started "
                f"{late} min late ({now_et():%H:%M} ET). GitHub queued or "
                f"delayed it, so today's coverage is shorter than usual.")

    if SESSION == "am":
        discord(f"🌅 **{today} — AM session live.** Watching {len(WATCHLIST)} "
                f"names, rescanning every 30 min until 12:40, then the PM "
                f"session takes over until 16:05. {paper_status()}")
    else:
        discord(f"🌇 **PM session live** — carrying {len(active)} setup(s) "
                f"from this morning, still rescanning every 30 min "
                f"until 16:05.")

    end_mins = hm(end)
    last_scan = 0.0
    while mins_now() < end_mins:
        if time.time() - last_scan >= RESCAN_EVERY:
            last_scan = time.time()
            if mins_now() < hm(NO_NEW_AFTER):
                try:
                    fresh = rescan(seen, len([c for c in active if not c["invalidated"]]))
                except Exception as e:
                    fresh = []
                    print("rescan error:", e)
                cards, skips = [], []
                for c in fresh:
                    if "contract" in c:
                        active.append(c)
                        cards.append(build_card(c, c["contract"],
                                                c.get("earn_note", "")))
                        log_candidate(c)
                    else:
                        skips.append(f"⏭️ {c['ticker']} {c['direction']} "
                                     f"setup found but skipped: {c['reject']}.")
                if cards or skips:
                    stamp = now_et().strftime("%H:%M ET")
                    discord(f"🎯 **Scan {stamp}** — {len(cards)} setup(s). "
                            f"Delayed data; verify at your broker.")
                    for card in cards:
                        discord(card)
                    if skips:
                        discord("\n".join(skips))
        try:
            poll(active)
        except Exception as e:
            print("poll error:", e)
        time.sleep(CHECK_EVERY)

    for c in active:
        log_candidate(c)

    if SESSION == "am":
        save_state(active)
        print("AM done; state handed to PM.")
    else:
        live = [c for c in active if c["fired"] and not c["invalidated"]]
        if live:
            discord("🔔 **16:05 — close.** You still show "
                    f"{len(live)} triggered setup(s) open. Anything held "
                    f"overnight risks a gap, and a gap can cost the full "
                    f"premium no matter where your stop sits. Decide "
                    f"deliberately, not by default.")
        else:
            discord("🌙 **16:05 — close.** Nothing left open. "
                    "Quiet days cost nothing; forced trades do.")
        try:
            os.remove(STATE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
