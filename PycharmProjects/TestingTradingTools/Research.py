"""
PART 1: Do EMA, RSI and ATR have an edge on XAUUSD?

QUESTION
    Does adding an indicator filter produce better trades than entering with
    no filter at all, using identical exits?

METHOD
    1. Baseline: enter on every free bar with the same SL/TP and costs.
       This is the "no indicator" reference. Any real edge must beat it.
    2. Test 7 rule sets per direction: EMA, RSI, ATR, and all combinations.
    3. Choose each rule's thresholds on the TRAIN segment only (first 60%).
    4. Report VALIDATION (next 20%) and HOLDOUT (last 20%) untouched by tuning.
    5. Pool validation + holdout into one "out-of-sample" (OOS) sample and
       test it with a t-statistic.
    6. Run every test twice: NET (with costs) and GROSS (zero costs).
       GROSS asks "does the indicator predict anything?".
       NET asks "can you actually make money from it?".

WHAT THE INDICATORS MEAN HERE
    EMA : trend filter. LONG = EMA20 > EMA50 > EMA200, SHORT = the reverse.
    RSI : pullback filter. LONG = RSI(14) <= threshold (oversold),
          SHORT = RSI(14) >= threshold (overbought).
    ATR : volatility-expansion filter. ATR / its 20-bar average >= threshold.

UNIT OF MEASUREMENT
    Results are in R-multiples (1R = the initial risk of the trade).
    ExpectancyR = average profit per trade in units of risk.

Make Sure to use Python 3.11 :) .
"""

import itertools
from pathlib import Path

import MetaTrader5 as mt5

path = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # adjust if needed

if not mt5.initialize(path=path):
    print("INIT:", mt5.last_error())
else:
    print("CONNECTED")
    print(mt5.terminal_info())
    print(mt5.account_info())
    mt5.shutdown()


import numpy as np
import pandas as pd

try:
    from numba import njit #njit is a decorator that compiles Python functions to optimized machine code using LLVM
except ImportError:
    raise SystemExit("Install Numba first: pip install numba")


# ----------------------------- CONFIG -----------------------------

SYMBOL = "XAUUSD"
OUTPUT_DIR = Path("xauusd_part1")

# Timeframes to test and how many bars to download for each.
TIMEFRAMES = {"5m": mt5.TIMEFRAME_M5, "4h": mt5.TIMEFRAME_H4}
BARS = {"5m": 500_000, "4h": 50_000}

# Chronological split. Thresholds are chosen on TRAIN only.
# VALIDATION and HOLDOUT are never used for selection.
TRAIN_FRAC = 0.60
VALIDATION_FRAC = 0.20          # holdout = the remaining 20%

# A rule needs at least this many trades in a segment to be trusted.
MIN_TRADES = {"5m": 100, "4h": 30}

# Maximum bars a trade may stay open before being closed at market.
MAX_BARS_HELD = {"5m": 288, "4h": 60}

# Round-trip cost in price units per ounce (spread + commission + slippage).
# Half is paid on entry, half on exit. Replace with your broker's real value.
# NOTE: with a 1-ATR stop on 5m, 0.50 can be a big fraction of 1R.
ROUND_TRIP_COST_PRICE = 0.50

# Fixed exits for ALL rules, so only the entry filter differs between them.
SL_ATR = 1.0                    # stop distance = 1.0 * ATR
RR = 1.5                        # target distance = 1.5 * stop distance

# Parameter grids. Kept small on purpose: fewer tries means less overfitting.
RSI_GRID = {"LONG": [20, 25, 30, 35, 40], "SHORT": [60, 65, 70, 75, 80]}
ATR_GRID = [1.0, 1.25, 1.5, 2.0]

# Minimum t-statistic on the out-of-sample trades to call something an edge.
# 2.0 is the usual 5% level. 2.5 is stricter because we test several rules.
T_MIN = 2.5


# ----------------------------- DATA & INDICATORS -----------------------------

def load_rates(timeframe, n_bars):
    """Download bars from MT5 and drop the still-forming candle."""
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates for {SYMBOL}; MT5 error={mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    # Position 0 is the live, incomplete bar. Remove it so results are stable.
    return df.iloc[:-1].reset_index(drop=True)


def add_indicators(df):
    """Compute EMA20/50/200, Wilder RSI(14), Wilder ATR(14) and ATR ratio."""
    df = df.copy()
    close = df["close"]

    # Exponential moving averages (recursive formula, like TradingView/MT5).
    for n in (20, 50, 200):
        df[f"EMA{n}"] = close.ewm(span=n, adjust=False, min_periods=n).mean()

    # Wilder RSI: average gain / average loss, smoothed with alpha = 1/14.
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["RSI"] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder ATR: smoothed true range (largest of the three range measures).
    prev_close = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # ATR ratio > 1 means volatility is above its recent average (expanding).
    df["ATR_RATIO"] = df["ATR"] / df["ATR"].rolling(20, min_periods=20).mean()

    # A bar is usable only when every indicator has warmed up.
    cols = ["EMA20", "EMA50", "EMA200", "RSI", "ATR", "ATR_RATIO"]
    df["valid"] = df[cols].notna().all(axis=1)
    return df


def to_arrays(df, tf):
    """Convert to plain numpy arrays: Numba needs arrays, not DataFrames."""
    f = lambda c: df[c].to_numpy(dtype=np.float64)
    return {
        "n": len(df), "tf": tf, "time": df["time"].to_numpy(),
        "open": f("open"), "high": f("high"), "low": f("low"), "close": f("close"),
        "atr": f("ATR"), "rsi": f("RSI"), "atr_ratio": f("ATR_RATIO"),
        "up": ((df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"])).to_numpy(),
        "down": ((df["EMA20"] < df["EMA50"]) & (df["EMA50"] < df["EMA200"])).to_numpy(),
        "valid": df["valid"].to_numpy(dtype=np.bool_),
    }


# ----------------------------- SIGNALS -----------------------------

def build_signal(a, direction, rule, lo, hi):
    """
    Boolean array: True where ALL indicators in `rule` agree.
    `rule` is a dict such as {"EMA": True, "RSI": 30, "ATR": 1.5}.
    A signal on bar i is known at bar i's CLOSE.
    """
    long_side = direction == "LONG"
    sig = a["valid"][lo:hi].copy()
    if "EMA" in rule:
        sig &= a["up" if long_side else "down"][lo:hi]
    if "RSI" in rule:
        rsi = a["rsi"][lo:hi]
        sig &= (rsi <= rule["RSI"]) if long_side else (rsi >= rule["RSI"])
    if "ATR" in rule:
        sig &= a["atr_ratio"][lo:hi] >= rule["ATR"]
    return sig


# ----------------------------- SIMULATOR -----------------------------

@njit(cache=True)
def simulate(open_, high, low, close, atr, entries, side, sl_mult, rr,
             max_bars, cost):
    """
    Simulate trades one at a time (no overlapping positions).

    - Signal on bar s (known at close) -> entry at the OPEN of bar s+1.
    - Stop size uses ATR at bar s (known at signal time, so no look-ahead).
    - Costs are applied adversely: half on entry, half on exit.
    - If SL and TP are both touched in one bar, the stop wins (pessimistic).
    - If price gaps through the stop, the exit is the gap open (realistic).
    - After max_bars the trade is closed at that bar's close.
    Returns the R-multiple of each completed trade.
    """
    n = len(close)
    out = np.empty(len(entries), dtype=np.float64)
    count = 0
    busy_until = -1
    half = cost / 2.0

    for k in range(len(entries)):
        s = entries[k]
        e = s + 1
        if e >= n or s <= busy_until:
            continue
        a = atr[s]
        if not (a > 0.0):
            continue

        entry = open_[e] + half if side == 1 else open_[e] - half
        risk = a * sl_mult
        sl = entry - risk if side == 1 else entry + risk
        tp = entry + risk * rr if side == 1 else entry - risk * rr

        last = min(n - 1, e + max_bars - 1)
        raw_exit = close[last]
        exit_i = last
        for j in range(e, last + 1):
            if side == 1:
                if low[j] <= sl:
                    raw_exit = open_[j] if open_[j] < sl else sl
                    exit_i = j
                    break
                if high[j] >= tp:
                    raw_exit = tp
                    exit_i = j
                    break
            else:
                if high[j] >= sl:
                    raw_exit = open_[j] if open_[j] > sl else sl
                    exit_i = j
                    break
                if low[j] <= tp:
                    raw_exit = tp
                    exit_i = j
                    break

        exit_px = raw_exit - half if side == 1 else raw_exit + half
        out[count] = side * (exit_px - entry) / risk
        count += 1
        busy_until = exit_i

    return out[:count]


def get_r(a, direction, rule, lo, hi, cost):
    """R-multiples for one rule on bars [lo, hi). rule=None means baseline."""
    if hi - lo < 3:
        return np.array([])
    if rule is None:
        # Baseline: try to enter on EVERY valid bar (skipped while in a trade).
        entries = np.flatnonzero(a["valid"][lo:hi])
    else:
        # Filtered: enter when the condition switches from False to True.
        sig = build_signal(a, direction, rule, lo, hi)
        entries = np.flatnonzero(np.diff(sig.astype(np.int8), prepend=0) == 1)
    if len(entries) == 0:
        return np.array([])
    s = slice(lo, hi)
    return simulate(a["open"][s], a["high"][s], a["low"][s], a["close"][s],
                    a["atr"][s], entries, 1 if direction == "LONG" else -1,
                    SL_ATR, RR, MAX_BARS_HELD[a["tf"]], float(cost))


def summarize(r):
    """Trades, expectancy (avg R), t-statistic, win rate, profit factor, DD."""
    if len(r) < 2:
        return {"n": len(r), "E": np.nan, "t": np.nan, "win": np.nan,
                "pf": np.nan, "dd": np.nan}
    mean, sd = r.mean(), r.std(ddof=1)
    t = mean / (sd / np.sqrt(len(r))) if sd > 0 else np.nan
    gross_loss = -r[r < 0].sum()
    pf = r[r > 0].sum() / gross_loss if gross_loss > 0 else np.nan
    curve = np.cumsum(r)                       # equity in R units
    dd = (np.maximum.accumulate(curve) - curve).max()
    return {"n": len(r), "E": mean, "t": t, "win": (r > 0).mean() * 100,
            "pf": pf, "dd": dd}


# ----------------------------- RULE SEARCH -----------------------------

def rule_variants(direction, subset):
    """All parameter combinations for a set of indicators, e.g. ('RSI','ATR')."""
    lists = {}
    if "EMA" in subset:
        lists["EMA"] = [True]
    if "RSI" in subset:
        lists["RSI"] = RSI_GRID[direction]
    if "ATR" in subset:
        lists["ATR"] = ATR_GRID
    names = list(lists)
    return [dict(zip(names, v)) for v in itertools.product(*(lists[n] for n in names))]


def rule_text(rule, direction):
    if rule is None:
        return "BASELINE (no filter)"
    parts = []
    if "EMA" in rule:
        parts.append("EMA stack")
    if "RSI" in rule:
        parts.append(f"RSI{'<=' if direction == 'LONG' else '>='}{rule['RSI']:g}")
    if "ATR" in rule:
        parts.append(f"ATR>={rule['ATR']:g}")
    return " + ".join(parts)


def evaluate_rule(a, direction, rule, bounds):
    """Report one rule on every segment, NET and GROSS."""
    tr_end, va_end, n = bounds
    cost = ROUND_TRIP_COST_PRICE
    train = summarize(get_r(a, direction, rule, 0, tr_end, cost))
    valid = summarize(get_r(a, direction, rule, tr_end, va_end, cost))
    hold = summarize(get_r(a, direction, rule, va_end, n, cost))

    # Pool validation + holdout for a larger out-of-sample sample.
    # (Pooling the R arrays is fine: mean and t-stat ignore trade order.)
    oos_net = summarize(np.concatenate([get_r(a, direction, rule, tr_end, va_end, cost),
                                        get_r(a, direction, rule, va_end, n, cost)]))
    oos_gross = summarize(np.concatenate([get_r(a, direction, rule, tr_end, va_end, 0.0),
                                          get_r(a, direction, rule, va_end, n, 0.0)]))
    return train, valid, hold, oos_net, oos_gross


def verdict(valid, hold, oos_net, oos_gross, base_gross_E):
    """
    EDGE   : positive in BOTH validation and holdout after costs, with a
             significant pooled t-stat.
    WEAK   : predictive before costs (and better than baseline) but costs
             remove the profit. The indicator "knows something" but you cannot
             trade it at this cost level.
    NO EDGE: neither of the above.
    """
    if (valid["E"] > 0 and hold["E"] > 0
            and oos_net["E"] > 0 and oos_net["t"] >= T_MIN):
        return "EDGE (survives costs)"
    if (oos_gross["E"] > 0 and oos_gross["t"] >= T_MIN
            and oos_gross["E"] > base_gross_E):
        return "WEAK (dies after costs)"
    return "NO EDGE"


def pipeline(tf, a):
    n = a["n"]
    bounds = (int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VALIDATION_FRAC)), n)
    tr_end = bounds[0]
    rows = []

    for direction in ("LONG", "SHORT"):
        # Baseline first: every other rule is compared against it.
        b = evaluate_rule(a, direction, None, bounds)
        base_gross_E = b[4]["E"]
        rows.append(make_row(tf, direction, None, b, "BASELINE", np.nan))

        # Every non-empty combination of the three indicators.
        for k in (1, 2, 3):
            for subset in itertools.combinations(["EMA", "RSI", "ATR"], k):
                best, best_E = None, -np.inf
                for rule in rule_variants(direction, subset):
                    r = get_r(a, direction, rule, 0, tr_end, ROUND_TRIP_COST_PRICE)
                    if len(r) < MIN_TRADES[tf]:
                        continue
                    # Choose thresholds on TRAIN only, by net expectancy.
                    if r.mean() > best_E:
                        best, best_E = rule, r.mean()
                if best is None:
                    print(f"[{tf}] {direction} {'+'.join(subset)}: too few trades")
                    continue
                res = evaluate_rule(a, direction, best, bounds)
                v = verdict(res[1], res[2], res[3], res[4], base_gross_E)
                lift = res[4]["E"] - base_gross_E
                rows.append(make_row(tf, direction, best, res, v, lift))
    return pd.DataFrame(rows)


def make_row(tf, direction, rule, res, verdict_text, lift):
    train, valid, hold, oos_net, oos_gross = res
    return {
        "TF": tf, "Dir": direction, "Rule": rule_text(rule, direction),
        "N_ind": 0 if rule is None else len(rule),
        "Train_n": train["n"], "Train_E": train["E"],
        "Valid_n": valid["n"], "Valid_E": valid["E"],
        "Hold_n": hold["n"], "Hold_E": hold["E"],
        "OOS_n": oos_net["n"], "OOS_E_net": oos_net["E"], "OOS_t_net": oos_net["t"],
        "OOS_E_gross": oos_gross["E"], "OOS_t_gross": oos_gross["t"],
        "Lift_vs_base": lift, "Verdict": verdict_text,
    }


# ----------------------------- MAIN -----------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        arrays = {}
        for name, tf in TIMEFRAMES.items():
            df = add_indicators(load_rates(tf, BARS[name]))
            arrays[name] = to_arrays(df, name)
            print(f"{name}: {len(df):,} bars | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    finally:
        mt5.shutdown()

    frames = [pipeline(name, arrays[name]) for name in TIMEFRAMES]
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(OUTPUT_DIR / "part1_results.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n=== FULL RESULTS (E = expectancy in R per trade) ===")
    print(results.round(3).to_string(index=False))

    # Automatic conclusion: look only at SINGLE-indicator rules, because they
    # answer "does this indicator alone have an edge?".
    print("\n=== SINGLE-INDICATOR CONCLUSION ===")
    single = results[results["N_ind"] == 1]
    for _, r in single.iterrows():
        print(f"[{r['TF']}] {r['Dir']:5s} {r['Rule']:18s} -> {r['Verdict']} "
              f"(OOS net E={r['OOS_E_net']:.3f}R, t={r['OOS_t_net']:.2f}; "
              f"gross E={r['OOS_E_gross']:.3f}R, lift vs baseline={r['Lift_vs_base']:.3f}R)")
    print(f"\nSaved: {(OUTPUT_DIR / 'part1_results.csv').resolve()}")


if __name__ == "__main__":
    main()


# ----------------------------- CONCLUSION -----------------------------
# Under the tested framework, EMA trend alignment, RSI pullback thresholds,
# ATR volatility-expansion filters, and their combinations
# did not demonstrate a statistically significant out-of-sample edge on XAUUSD.

# ----------------------------- POSSIBLE ERRORS -----------------------------
# 1) If you install a package (e.g. Numba) but the script still says:
#    "Install Numba first: pip install numba"
#
#    you are probably installing into a different virtual environment than
#    the one used to run the script.
#
#    Check which interpreter PyCharm is using and install the package into
#    that exact environment. For example:
#
#    .\.venv1\Scripts\activate
#    pip install numba
#
#    or:
#
#    .\.venv1\Scripts\python.exe -m pip install numba
