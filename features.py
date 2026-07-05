import math

def _safe_pct_change(current, previous):
    if previous is None or abs(previous) < 1e-12:
        return 0.0
    return (current / previous - 1.0) * 100.0


def _sma(values, period):
    if len(values) < period:
        return [None] * len(values)
    result = [None] * (period - 1)
    for i in range(period - 1, len(values)):
        result.append(sum(values[i - period + 1: i + 1]) / period)
    return result


def _ema(values, period):
    if len(values) < period:
        return [None] * len(values)
    k = 2.0 / (period + 1)
    result = [None] * (period - 1)
    result.append(sum(values[:period]) / period)   # seed with SMA
    for i in range(period, len(values)):
        result.append(values[i] * k + result[-1] * (1 - k))
    return result


def _rsi(closes, period=14):
    n = len(closes)
    if n <= period:
        return [None] * n

    result = [None] * period

    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def _rs_to_rsi(ag, al):
        if al == 0:
            return 100.0
        return 100.0 - 100.0 / (1 + ag / al)

    result.append(_rs_to_rsi(avg_gain, avg_loss))

    for i in range(period + 1, n):
        diff     = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0))  / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0)) / period
        result.append(_rs_to_rsi(avg_gain, avg_loss))

    return result


def _macd_histogram(closes, fast=12, slow=26, signal=9):
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)

    # MACD line (valid where both EMAs are valid, i.e. from index slow-1 onwards)
    macd_line = []
    for f, s in zip(ema_fast, ema_slow):
        macd_line.append(None if (f is None or s is None) else f - s)

    # Signal line = EMA(9) of the valid MACD values
    valid_start = next((i for i, v in enumerate(macd_line) if v is not None), None)
    if valid_start is None:
        empty = [None] * len(closes)
        return empty, empty, empty

    valid_macd  = [v for v in macd_line if v is not None]
    ema_signal  = _ema(valid_macd, signal)

    # Map signal values back to full-length list
    signal_line = [None] * len(closes)
    for idx, orig_idx in enumerate(
        [i for i, v in enumerate(macd_line) if v is not None]
    ):
        signal_line[orig_idx] = ema_signal[idx]

    histogram = []
    for m, s in zip(macd_line, signal_line):
        histogram.append(None if (m is None or s is None) else m - s)

    return histogram, macd_line, signal_line


def _bollinger(closes, period=20, n_std=2):
    """Returns (upper, middle/SMA, lower, %B position)."""
    sma    = _sma(closes, period)
    upper  = [None] * len(closes)
    lower  = [None] * len(closes)
    pct_b  = [None] * len(closes)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1: i + 1]
        mean   = sma[i]
        std    = math.sqrt(sum((x - mean) ** 2 for x in window) / period)
        u = mean + n_std * std
        l = mean - n_std * std
        upper[i] = u
        lower[i] = l
        pct_b[i] = (closes[i] - l) / (u - l) if (u - l) > 1e-9 else 0.5

    return upper, sma, lower, pct_b


def _volatility(closes, period=10):
    returns = [None]
    for i in range(1, len(closes)):
        r = _safe_pct_change(closes[i], closes[i - 1])
        returns.append(r)

    if len(closes) < period:
        return [None] * len(closes)

    result = [None] * period
    for i in range(period, len(closes)):
        window = returns[i - period + 1: i + 1]
        mean   = sum(window) / period
        std    = math.sqrt(sum((x - mean) ** 2 for x in window) / period)
        result.append(round(std, 6))

    return result


FEATURE_NAMES = [
    "RSI (14)",
    "SMA10/SMA50 Ratio",
    "MACD Histogram",
    "Bollinger %B",
    "Volume Change %",
    "Price Change 1d %",
    "Price Change 3d %",
    "Price Change 5d %",
    "Volatility 10d",
]


def _compute_indicators(closes):
    rsi14          = _rsi(closes, 14)
    sma10          = _sma(closes, 10)
    sma50          = _sma(closes, 50)
    macd_hist, macd_line, signal_line = _macd_histogram(closes)
    bb_upper, bb_mid, bb_lower, pct_b = _bollinger(closes, 20)
    vol10          = _volatility(closes, 10)

    return {
        "rsi14"      : rsi14,
        "sma10"      : sma10,
        "sma50"      : sma50,
        "macd_hist"  : macd_hist,
        "macd_line"  : macd_line,
        "signal_line": signal_line,
        "bb_upper"   : bb_upper,
        "bb_mid"     : bb_mid,
        "bb_lower"   : bb_lower,
        "pct_b"      : pct_b,
        "vol10"      : vol10,
    }


def _feature_row(i, closes, volumes, indicators):
    if i < 5 or i >= len(closes):
        return None

    rsi14     = indicators["rsi14"]
    sma10     = indicators["sma10"]
    sma50     = indicators["sma50"]
    macd_hist = indicators["macd_hist"]
    pct_b     = indicators["pct_b"]
    vol10     = indicators["vol10"]

    if i >= len(volumes):
        return None

    if any(v is None for v in [
        rsi14[i], sma10[i], sma50[i],
        macd_hist[i], pct_b[i], vol10[i],
    ]):
        return None

    sma_ratio = _safe_pct_change(sma10[i], sma50[i])
    vol_chg   = _safe_pct_change(volumes[i], volumes[i - 1])
    pc1       = _safe_pct_change(closes[i], closes[i - 1])
    pc3       = _safe_pct_change(closes[i], closes[i - 3])
    pc5       = _safe_pct_change(closes[i], closes[i - 5])

    return [
        round(rsi14[i],     4),
        round(sma_ratio,    4),
        round(macd_hist[i], 6),
        round(pct_b[i],     4),
        round(vol_chg,      4),
        round(pc1,          4),
        round(pc3,          4),
        round(pc5,          4),
        round(vol10[i],     6),
    ]


def build_features(dates, closes, volumes):
    n = len(closes)
    indicators = _compute_indicators(closes)

    X, y, X_dates = [], [], []

    for i in range(5, n - 1):
        row = _feature_row(i, closes, volumes, indicators)
        if row is None:
            continue

        label = 1 if closes[i + 1] > closes[i] else 0

        X.append(row)
        y.append(label)
        X_dates.append(dates[i])

    raw = {
        "sma10"      : indicators["sma10"],
        "sma50"      : indicators["sma50"],
        "bb_upper"   : indicators["bb_upper"],
        "bb_lower"   : indicators["bb_lower"],
        "macd_line"  : indicators["macd_line"],
        "signal_line": indicators["signal_line"],
        "macd_hist"  : indicators["macd_hist"],
        "rsi14"      : indicators["rsi14"],
    }

    return X, y, X_dates, raw


def build_latest_features(dates, closes, volumes):
    indicators = _compute_indicators(closes)
    row = _feature_row(len(closes) - 1, closes, volumes, indicators)
    if row is None:
        return None, None
    return row, dates[-1]


def current_indicators(closes, volumes, feature_names=None):
    row, _ = build_latest_features(
        [str(i) for i in range(len(closes))], closes, volumes
    )
    return row
