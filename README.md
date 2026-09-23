## Multi-Ticker Results

| Ticker | Model MAPE | Naive MAPE | Delta   |
| ------ | ---------- | ---------- | ------- |
| AAPL   | 1.12%      | 1.13%      | +0.01%  |
| MSFT   | 1.42%      | 1.41%      | −0.01% |
| TSLA   | 2.21%      | 2.23%      | +0.02%  |
| SPY    | 0.64%      | 0.64%      | +0.00%  |

All runs use ~4 years of training data and ~1 year of held out test data,
with a 60-day lookback window, a 2-layer LSTM (64 hidden units) and a
residual connection to the last observed close.

### Interpretation

Every model sits within 0.02% MAPE of the naive baseline (predict
tomorrow = today). The model beats the baseline on two tickers, loses on
one and ties on the fourth but all deltas are within what you'd expect
from random seed variation. **The model is statistically indistinguishable
from the trivial baseline.**

This is the expected result. Daily stock prices follow a near random walk.
The naive baseline is famously hard to beat because it captures the
strong autocorrelation in daily closing prices, tomorrow's price is
usually within 1% of today's.

The variation across tickers is instructive:

- **SPY** (0.64% MAPE): The S&P 500 ETF is highly efficient and "no
  change" is nearly optimal. The model matches it exactly.
- **TSLA** (2.21% MAPE): Higher day-to-day volatility means higher error
  for both model and baseline, with no predictive edge gained.
- **AAPL / MSFT** (~1.1–1.4% MAPE): Typical large-cap behavior. The
  ±0.01% deltas are noise not signal.

### Why this is the honest result

Most "stock predictor" repositories report 90%+ accuracy or dramatic
outperformance. Those claims usually stem from data leakage (shuffling
time series), reporting training metrics as test metrics or not
comparing against a baseline.

This project compares against the correct baseline, uses a proper
chronological split, restores the best-validation state and reports
the truth: **an LSTM on daily OHLCV data cannot reliably beat the
naive "no change" baseline.** That is the state of the art for this
problem.

### What this project demonstrates

- A complete ML pipeline: data ingestion → feature engineering →
  sequence construction → LSTM training → evaluation
- Proper chronological train/test splitting (no leakage)
- Rigorous baseline comparison, not just model metrics
- Best-state restoration (early stopping)
- Multi-ticker validation
