# StockSense AI

StockSense AI is a Flask web app that fetches recent market data with `yfinance`, builds technical indicators, and trains an entropy-based decision tree to predict the next trading session's stock direction.

## What changed

- Redesigned the interface into a premium Apple-like dark console.
- Removed the fragile D3 CDN dependency and replaced the tree view with native SVG rendering.
- Fixed the live prediction bug so the next-session call uses the latest available close instead of the last labeled training row.
- Switched to a shallow decision tree that generalizes better across the bundled ticker list.
- Added walk-forward backtesting so recent results are measured with models trained only on prior rows.
- Added inline error handling, responsive layouts, metadata, favicon, and reproducible dependencies.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Notes

The predictions are educational model outputs, not financial advice. Short-horizon market direction is noisy; use the holdout and walk-forward metrics to judge each ticker's current behavior.
