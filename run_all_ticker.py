import numpy as np
import torch
import matplotlib.pyplot as plt

from stock_predictor import (
    fetch_data,
    preprocess,
    StockLSTM,
    train_model,
    evaluate,
    naive_baseline,
    LOOKBACK,
    TRAIN_SPLIT,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    PERIOD,
    device,
)

TICKERS = ["AAPL", "MSFT", "TSLA", "SPY"]


def run_one(ticker):
    """Run the full pipeline for a single ticker. Returns a dict of results."""
    print(f"\n{'=' * 60}")
    print(f"{ticker}")
    print(f"{'=' * 60}")

    df = fetch_data(ticker, PERIOD)
    X_train, y_train, X_test, y_test, scaler, close_idx = preprocess(
        df, LOOKBACK, TRAIN_SPLIT
    )

    n_features = X_train.shape[2]
    model = StockLSTM(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)

    train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test,
        EPOCHS, BATCH_SIZE, LEARNING_RATE,
    )

    preds, actual, model_mape = evaluate(
        model, X_test, y_test, scaler, close_idx, n_features
    )

    closes = df["Close"].values.astype(np.float32)
    n_samples = len(closes) - LOOKBACK
    split = int(n_samples * TRAIN_SPLIT)
    test_start = LOOKBACK + split
    naive_preds = closes[test_start - 1 : test_start - 1 + len(actual)]
    naive_actual = closes[test_start : test_start + len(actual)]

    model_rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((naive_preds - naive_actual) ** 2)))
    naive_mape = float(np.mean(np.abs((naive_preds - naive_actual) / naive_actual)) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses, label="Val")
    axes[0].set_title(f"{ticker} — Loss")
    axes[0].legend()
    axes[1].plot(actual, label="Actual", alpha=0.8)
    axes[1].plot(preds, label="Predicted", alpha=0.8)
    axes[1].set_title(f"{ticker} — Predicted vs Actual")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_prediction.png", dpi=120)
    plt.close()

    return {
        "ticker": ticker,
        "rows": len(df),
        "model_rmse": model_rmse,
        "model_mape": model_mape,
        "naive_rmse": naive_rmse,
        "naive_mape": naive_mape,
        "delta": naive_mape - model_mape,
    }


if __name__ == "__main__":
    results = []
    for ticker in TICKERS:
        try:
            results.append(run_one(ticker))
        except Exception as e:
            print(f"\n!! {ticker} failed: {e}\n")
            results.append({
                "ticker": ticker,
                "rows": 0,
                "model_rmse": float("nan"),
                "model_mape": float("nan"),
                "naive_rmse": float("nan"),
                "naive_mape": float("nan"),
                "delta": float("nan"),
            })

    #summary table
    print(f"\n\n{'=' * 78}")
    print("FINAL COMPARISON")
    print(f"{'=' * 78}")
    header = f"{'Ticker':<8} {'Rows':<6} {'Model RMSE':<12} {'Naive RMSE':<12} {'Model MAPE':<12} {'Naive MAPE':<12} {'Delta':<8}"
    print(header)
    print("-" * 78)
    for r in results:
        if np.isnan(r["model_mape"]):
            print(f"{r['ticker']:<8} {'FAILED':<6}")
            continue
        print(
            f"{r['ticker']:<8} {r['rows']:<6} "
            f"{r['model_rmse']:<12.4f} {r['naive_rmse']:<12.4f} "
            f"{r['model_mape']:<12.2f} {r['naive_mape']:<12.2f} "
            f"{r['delta']:+.2f}%"
        )
    print("=" * 78)
    print("Delta > 0 means the model beats the naive baseline")