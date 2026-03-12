#!/usr/bin/env python3
"""
STI–SMID Vol Rotation Signal Pipeline
--------------------------------------
Fetches STI daily data via yfinance, computes 25-day realised vol,
determines signal status, appends to signal log CSV, and bakes
data into docs/index.html for GitHub Pages.

Run: python scripts/pipeline.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ──
TICKER = "^STI"
VOL_WINDOW = 25
TRIGGER = 0.165        # 16.5% annualised
MEDIAN = 0.140         # 14.0%
APPROACHING_PCT = 0.85 # 85% of trigger = ~14.0%
RATE_THRESHOLD = 0.005 # 50bps

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "template" / "index.html"
OUTPUT = ROOT / "docs" / "index.html"
LOG_CSV = ROOT / "data" / "signal_log.csv"

# Vol percentile lookup (calibrated 2010-2025)
VOL_PCTS = [
    (5, 0.065), (10, 0.085), (15, 0.095), (20, 0.105),
    (25, 0.110), (30, 0.118), (35, 0.125), (40, 0.132),
    (45, 0.136), (50, 0.140), (55, 0.145), (60, 0.150),
    (65, 0.155), (70, 0.160), (75, 0.165), (80, 0.172),
    (85, 0.180), (90, 0.195), (95, 0.230), (100, 0.450),
]

def vol_to_percentile(v):
    for i, (p, threshold) in enumerate(VOL_PCTS):
        if v <= threshold:
            if i == 0:
                return p
            p0, v0 = VOL_PCTS[i - 1]
            return p0 + (p - p0) * (v - v0) / (threshold - v0)
    return 99


def fetch_sti(days=400):
    """Fetch STI daily OHLCV data."""
    end = datetime.now()
    start = end - timedelta(days=days)
    print(f"Fetching {TICKER} from {start.date()} to {end.date()}…")
    df = yf.download(TICKER, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        print("ERROR: No data returned from yfinance.")
        sys.exit(1)
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna().copy()
    df.columns = ["close"]
    print(f"  → {len(df)} trading days fetched. Latest: {df.index[-1].date()} @ {df['close'].iloc[-1]:.1f}")
    return df


def compute_vol(df):
    """Compute rolling 25-day annualised realised volatility."""
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_25d"] = df["log_ret"].rolling(VOL_WINDOW).std() * np.sqrt(252)
    df = df.dropna(subset=["vol_25d"]).copy()
    df["vol_pct"] = df["vol_25d"].apply(vol_to_percentile)
    return df


def determine_signal(current_vol, rate_override=False):
    """Determine signal status."""
    if current_vol >= TRIGGER and rate_override:
        return "OVERRIDE", "Rate-regime override active. Vol trigger met but SG 10Y yield spike detected."
    elif current_vol >= TRIGGER:
        return "TRIGGERED", f"25d vol at {current_vol*100:.1f}% exceeds {TRIGGER*100:.1f}% threshold. Initiate rotation."
    elif current_vol >= TRIGGER * APPROACHING_PCT:
        return "APPROACHING", f"25d vol at {current_vol*100:.1f}% — within 15% of trigger. Monitor closely."
    else:
        return "DORMANT", f"25d vol at {current_vol*100:.1f}% — well below trigger. No action required."


def append_log(df, signal, desc):
    """Append latest reading to signal log CSV."""
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    latest = df.iloc[-1]
    row = {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "sti_close": round(latest["close"], 2),
        "vol_25d": round(latest["vol_25d"] * 100, 2),
        "vol_pct": round(latest["vol_pct"], 0),
        "signal": signal,
        "description": desc,
        "rate_override": "N/A",
    }
    if LOG_CSV.exists():
        log = pd.read_csv(LOG_CSV)
        # Don't duplicate today
        if row["date"] not in log["date"].values:
            log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log.to_csv(LOG_CSV, index=False)
    print(f"  → Signal log updated: {len(log)} entries. Latest: {signal}")


def bake_html(df, signal, desc):
    """Inject precomputed data into HTML template and write to docs/."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE.read_text(encoding="utf-8")

    # Prepare JSON data for injection
    vol_data = []
    for idx, row in df.tail(300).iterrows():
        vol_data.append({
            "d": idx.strftime("%Y-%m-%d"),
            "c": round(row["close"], 2),
            "v": round(row["vol_25d"] * 100, 2),
            "p": round(row["vol_pct"], 1),
        })

    latest = df.iloc[-1]
    meta = {
        "signal": signal,
        "desc": desc,
        "vol": round(latest["vol_25d"] * 100, 1),
        "pct": round(latest["vol_pct"], 0),
        "sti": round(latest["close"], 1),
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "trigger": TRIGGER * 100,
        "median": MEDIAN * 100,
    }

    injection = f"""
<script>
// ── BAKED DATA (auto-generated by pipeline.py) ──
window.BAKED_DATA = {json.dumps(vol_data)};
window.BAKED_META = {json.dumps(meta)};
</script>
"""

    # Inject before closing </body>
    html = template.replace("</body>", injection + "</body>")
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"  → HTML baked to {OUTPUT} ({len(html):,} bytes)")


def main():
    print("=" * 60)
    print("STI–SMID Vol Rotation Signal Pipeline")
    print("=" * 60)

    df = fetch_sti()
    df = compute_vol(df)

    latest = df.iloc[-1]
    current_vol = latest["vol_25d"]
    current_pct = latest["vol_pct"]
    distance = TRIGGER - current_vol

    print(f"\n── Signal Status ──")
    print(f"  STI Close:    {latest['close']:.1f}")
    print(f"  25d Vol:      {current_vol*100:.1f}%")
    print(f"  Percentile:   {current_pct:.0f}th")
    print(f"  Distance:     {distance*100:+.1f}% to trigger")

    signal, desc = determine_signal(current_vol)
    print(f"  Signal:       {signal}")
    print(f"  Description:  {desc}\n")

    append_log(df, signal, desc)
    bake_html(df, signal, desc)

    print("\nDone.")


if __name__ == "__main__":
    main()
