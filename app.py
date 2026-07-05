from flask import Flask, render_template, request, jsonify
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

from features import build_features, build_latest_features, FEATURE_NAMES

app = Flask(__name__)

STOCKS = {
    "AAPL"       : "Apple Inc.",
    "TSLA"       : "Tesla Inc.",
    "GOOGL"      : "Alphabet Inc.",
    "MSFT"       : "Microsoft Corp.",
    "AMZN"       : "Amazon.com Inc.",
    "NVDA"       : "NVIDIA Corp.",
    "INFY"       : "Infosys Ltd.",
    "TCS.NS"     : "Tata Consultancy Services",
    "RELIANCE.NS": "Reliance Industries",
    "HDFCBANK.NS": "HDFC Bank",
}

DEFAULT_MODEL_PARAMS = {
    "max_depth": 2,
    "min_samples_split": 8,
    "min_samples_leaf": 3,
    "class_weight": None,
}


def _fetch(ticker):
    hist = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
    hist = hist.dropna(subset=["Close", "Volume"])
    if hist.empty:
        return None
    return {
        "dates"  : [str(d.date()) for d in hist.index],
        "closes" : [round(float(v), 4) for v in hist["Close"]],
        "volumes": [int(v) for v in hist["Volume"]],
    }


def _currency_symbol(ticker):
    return "\u20b9" if ticker.endswith(".NS") else "$"


def _make_model(params):
    return DecisionTreeClassifier(
        criterion="entropy",
        random_state=42,
        **params,
    )


def _select_model_params(X_train, y_train):
    """Use a shallow tree that tested more consistently across the bundled tickers."""
    params = DEFAULT_MODEL_PARAMS.copy()
    if len(X_train) < 120:
        return params, None

    val_size = max(35, int(len(X_train) * 0.20))
    split = len(X_train) - val_size
    if split < 70:
        return params, None

    model = _make_model(params)
    model.fit(X_train[:split], y_train[:split])
    validation_acc = round(model.score(X_train[split:], y_train[split:]) * 100, 1)
    return params, validation_acc


def _predict_one(model, x):
    """Walk the fitted tree for one sample; return prediction, confidence, path."""
    t, node, path = model.tree_, 0, []
    while t.children_left[node] != -1:
        fi  = int(t.feature[node])
        thr = float(t.threshold[node])
        val = float(x[fi])
        lft = bool(val <= thr)
        n   = int(t.n_node_samples[node])
        nl  = int(t.n_node_samples[t.children_left[node]])
        nr  = int(t.n_node_samples[t.children_right[node]])
        ig  = float(t.impurity[node]) - (
              (nl/n) * float(t.impurity[t.children_left[node]]) +
              (nr/n) * float(t.impurity[t.children_right[node]]))
        path.append({
            "feature"  : FEATURE_NAMES[fi],
            "value"    : round(val, 4),
            "threshold": round(thr, 4),
            "direction": f"\u2264 {round(thr,4)}" if lft else f"> {round(thr,4)}",
            "went_left": lft,
            "info_gain": round(ig, 6),
            "samples"  : n,
        })
        node = t.children_left[node] if lft else t.children_right[node]
    counts = t.value[node][0]
    total  = float(sum(counts)) or 1.0
    pred_i = int(counts.argmax())
    pred   = int(model.classes_[pred_i])
    return pred, round(float(counts[pred_i]) / total * 100, 1), path


def _tree_to_dict(model, node_id=0, depth=0):
    """Serialise sklearn tree to nested dict for D3."""
    t       = model.tree_
    counts  = t.value[node_id][0]
    total   = float(sum(counts)) or 1.0
    pred_i  = int(counts.argmax())
    pred    = int(model.classes_[pred_i])
    is_leaf = bool(t.children_left[node_id] == -1)
    d = {
        "samples"     : int(t.n_node_samples[node_id]),
        "prediction"  : pred,
        "probability" : round(float(counts[pred_i]) / total * 100, 1),
        "class_counts": {
            str(int(model.classes_[i])): round(float(c), 2)
            for i, c in enumerate(counts)
        },
        "is_leaf"     : is_leaf,
        "depth"       : depth,
    }
    if not is_leaf:
        fi  = int(t.feature[node_id])
        n   = int(t.n_node_samples[node_id])
        nl  = int(t.n_node_samples[t.children_left[node_id]])
        nr  = int(t.n_node_samples[t.children_right[node_id]])
        ig  = float(t.impurity[node_id]) - (
              (nl/n) * float(t.impurity[t.children_left[node_id]]) +
              (nr/n) * float(t.impurity[t.children_right[node_id]]))
        d["feature"]   = FEATURE_NAMES[fi]
        d["threshold"] = round(float(t.threshold[node_id]), 6)
        d["info_gain"] = round(ig, 6)
        d["left"]  = _tree_to_dict(model, int(t.children_left[node_id]),  depth + 1)
        d["right"] = _tree_to_dict(model, int(t.children_right[node_id]), depth + 1)
    return d


def _backtest(params, X, y, dates, n_days=90):
    """Walk-forward test: each day is predicted by a model trained only on prior days."""
    results = []
    start = max(70, len(X) - n_days)
    for idx in range(start, len(X)):
        model = _make_model(params)
        model.fit(X[:idx], y[:idx])
        xi, yi, di = X[idx], y[idx], dates[idx]
        pred, conf, _ = _predict_one(model, xi)
        results.append({
            "date"      : di,
            "predicted" : int(pred),
            "actual"    : int(yi),
            "correct"   : bool(pred == yi),
            "confidence": conf,
        })
    return results


@app.route("/")
def index():
    return render_template("index.html", stocks=STOCKS)

@app.route("/api/stocks")
def api_stocks():
    return jsonify(STOCKS)

@app.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json(silent=True) or {}
    ticker = payload.get("ticker", "AAPL").upper()

    raw = _fetch(ticker)
    if raw is None:
        return jsonify({"error": f"Could not fetch data for {ticker}"}), 400

    X, y, feat_dates, _ = build_features(raw["dates"], raw["closes"], raw["volumes"])
    latest_x, latest_feature_date = build_latest_features(
        raw["dates"], raw["closes"], raw["volumes"]
    )

    if len(X) < 50:
        return jsonify({"error": "Not enough data to train"}), 400
    if latest_x is None:
        return jsonify({"error": "Not enough recent indicator data to predict"}), 400

    split   = int(len(X) * 0.80)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    params, validation_acc = _select_model_params(X_train, y_train)

    eval_model = _make_model(params)
    eval_model.fit(X_train, y_train)

    train_acc = round(eval_model.score(X_train, y_train) * 100, 1)
    test_acc  = round(eval_model.score(X_test,  y_test)  * 100, 1)

    model = _make_model(params)
    model.fit(X, y)

    feat_imp = dict(sorted(
        {FEATURE_NAMES[i]: round(float(model.feature_importances_[i] * 100), 1)
         for i in range(len(FEATURE_NAMES))}.items(),
        key=lambda kv: -kv[1]))

    pred, confidence, path = _predict_one(model, latest_x)
    current_vals = {FEATURE_NAMES[i]: round(latest_x[i], 4) for i in range(len(latest_x))}

    backtest = _backtest(params, X, y, feat_dates)
    win_rate = round(sum(r["correct"] for r in backtest) / len(backtest) * 100, 1) if backtest else 0.0

    pnl = sum(200.0 if r["correct"] else -200.0 for r in backtest)

    return jsonify({
        "ticker"            : ticker,
        "company"           : STOCKS.get(ticker, ticker),
        "train_acc"         : train_acc,
        "test_acc"          : test_acc,
        "validation_acc"    : validation_acc,
        "n_train"           : len(X),
        "n_test"            : len(X_test),
        "feature_importance": feat_imp,
        "tree"              : _tree_to_dict(model),
        "prediction"        : int(pred),
        "confidence"        : confidence,
        "decision_path"     : path,
        "current_indicators": current_vals,
        "backtest"          : backtest,
        "win_rate"          : win_rate,
        "pnl_final"         : round(pnl, 2),
        "latest_close"      : raw["closes"][-1],
        "latest_date"       : raw["dates"][-1],
        "feature_date"      : latest_feature_date,
        "currency_symbol"   : _currency_symbol(ticker),
        "model_params"      : params,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
